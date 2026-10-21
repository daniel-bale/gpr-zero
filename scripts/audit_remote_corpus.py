#!/usr/bin/env python3
"""Audit saved thesis corpus metadata on bdsrc without modifying remote files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess


REMOTE_CODE = r"""
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
import duckdb

RUN_ORIGINAL_BINARY_COUNTS = __RUN_ORIGINAL_BINARY_COUNTS__
db_path = Path('/home5/chuenho/data/news/stage2_enhanced_full.duckdb')
con = duckdb.connect(str(db_path), read_only=True)
con.execute("SET threads=2")
con.execute("SET memory_limit='512MB'")
con.execute("SET max_temp_directory_size='0B'")
result = {
    'python_executable': sys.executable,
    'duckdb_version': duckdb.__version__,
    'database': str(db_path),
    'database_bytes': db_path.stat().st_size,
    'read_only': True,
    'query_timeout_seconds': 45,
    'query_limits': {'threads': 2, 'memory_limit': '512MB', 'temporary_disk_limit': '0B'},
    'queries': {},
}

def query(name, sql):
    start = time.monotonic()
    timer = threading.Timer(45, con.interrupt)
    timer.daemon = True
    timer.start()
    try:
        cursor = con.execute(sql)
        columns = [item[0] for item in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        result['queries'][name] = {'sql': sql, 'rows': rows, 'seconds': round(time.monotonic() - start, 3)}
        return rows
    except Exception as error:
        result['queries'][name] = {'sql': sql, 'error': str(error), 'seconds': round(time.monotonic() - start, 3)}
        return None
    finally:
        timer.cancel()

tables = query('tables', "SELECT table_name, estimated_size AS estimated_rows, column_count FROM duckdb_tables() WHERE NOT internal ORDER BY table_name")
query('views', "SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY view_name")
names = {row['table_name'] for row in (tables or [])}
table = 'gpr_binary_matrix_all_words' if 'gpr_binary_matrix_all_words' in names else 'articles'
result['metadata_count_source'] = table
query('metadata_schema', "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema='main' AND column_name NOT LIKE 'contains_word_%' ORDER BY table_name, ordinal_position")
query('binary_column_summary', "SELECT count(*) AS binary_columns, min(column_name) AS first_binary_column, max(column_name) AS last_binary_column FROM information_schema.columns WHERE table_name='gpr_binary_matrix_all_words' AND column_name LIKE 'contains_word_%'")
query('corpus_total', f'SELECT count(*) AS total_rows, min(warc_date) AS first_warc_date, max(warc_date) AS last_warc_date FROM "{table}"')
query('three_outlet_counts_2017_2024', f'''SELECT outlet, count(*) AS article_count FROM "{table}" WHERE warc_date >= TIMESTAMP '2017-01-01' AND warc_date < TIMESTAMP '2025-01-01' AND outlet IN ('China Times (中國時報)', 'Liberty Times (自由時報)', 'United Daily (聯合報)') GROUP BY outlet ORDER BY outlet''')
query('three_outlet_year_counts_2017_2024', f'''SELECT extract(year FROM warc_date)::INTEGER AS year, outlet, count(*) AS article_count FROM "{table}" WHERE warc_date >= TIMESTAMP '2017-01-01' AND warc_date < TIMESTAMP '2025-01-01' AND outlet IN ('China Times (中國時報)', 'Liberty Times (自由時報)', 'United Daily (聯合報)') GROUP BY year, outlet ORDER BY year, outlet''')

dictionary_dir = Path('/home5/chuenho/gpr-zero/data/dictionaries/all_dictionaries')
all_words = set()
dictionaries = []
for path in sorted(dictionary_dir.glob('*.json')):
    raw = path.read_bytes()
    data = json.loads(raw)
    if 'word_sets' not in data or 'categories' not in data:
        continue
    words = {word for values in data.get('word_sets', {}).values() for word in values}
    all_words.update(words)
    dictionaries.append({'filename': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'word_count': len(words), 'word_set_count': len(data.get('word_sets', {})), 'category_count': len(data.get('categories', {}))})
result['dictionary_manifest'] = {'directory': str(dictionary_dir), 'count': len(dictionaries), 'unique_word_count': len(all_words), 'sorted_words_sha256': hashlib.sha256(json.dumps(sorted(all_words), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(), 'files': dictionaries}
result['president_and_lau_candidates'] = [item for item in dictionaries if 'president' in item['filename'].lower() or item['filename'].lower().startswith(('lau_', 'yeu_'))]

def identifier(value):
    return '"' + value.replace('"', '""') + '"'

target_outlets = {'China Times (中國時報)', 'Liberty Times (自由時報)', 'United Daily (聯合報)'}
president_claims = {'Military Escalation': 33.48, 'Internal Political Instability': 32.90, 'Diplomatic Crisis': 19.05, 'Foreign Intervention Risk': 12.35, 'Economic Coercion': 10.81, 'Cyber and Information Warfare': 10.22, 'Critical Infrastructure Threat': 0.89}

def dictionary_audit(name, dictionary_path, mapping_path, source_table, expected_union=None):
    dictionary_bytes = dictionary_path.read_bytes()
    dictionary = json.loads(dictionary_bytes)
    mapping_bytes = mapping_path.read_bytes()
    mapping = json.loads(mapping_bytes)
    words = {word for values in dictionary['word_sets'].values() for word in values}
    cached_columns = {row[0] for row in con.execute('SELECT column_name FROM information_schema.columns WHERE table_name = ?', [source_table]).fetchall() if row[0].startswith('contains_word_')}
    missing_words = sorted(words - mapping.keys())
    details = {
        'dictionary_path': str(dictionary_path), 'dictionary_sha256': hashlib.sha256(dictionary_bytes).hexdigest(),
        'mapping_path': str(mapping_path), 'mapping_sha256': hashlib.sha256(mapping_bytes).hexdigest(),
        'source_table': source_table, 'dictionary_words': len(words), 'mapping_words': len(mapping),
        'binary_columns': len(cached_columns), 'missing_dictionary_words': missing_words,
        'mapping_values_match_binary_schema': set(mapping.values()) == cached_columns,
        'mapping_is_one_to_one': len(set(mapping.values())) == len(mapping),
    }
    if expected_union is not None:
        expected = {word: f'contains_word_{index:04d}' for index, word in enumerate(sorted(expected_union))}
        details['mapping_equals_sorted_union'] = mapping == expected
    result[name] = details
    if missing_words or not details['mapping_values_match_binary_schema'] or not details['mapping_is_one_to_one'] or details.get('mapping_equals_sorted_union') is False:
        details['error'] = 'Mapping validation failed; no category aggregation attempted.'
        return
    if name == 'original_president' and not RUN_ORIGINAL_BINARY_COUNTS:
        details['category_query_status'] = 'Full binary scan skipped; use bounded labeled-cache reconstruction below.'
        return
    category_expressions = []
    category_names = list(dictionary['categories'])
    for category in category_names:
        set_names = [part.strip() for part in re.split(r'\s+AND\s+|\+', dictionary['categories'][category])]
        if any(set_name not in dictionary['word_sets'] for set_name in set_names):
            details['error'] = f'Unknown word-set name in category {category}; no aggregation attempted.'
            return
        set_expressions = []
        for set_name in set_names:
            predicates = [f'coalesce({identifier(mapping[word])}, FALSE)' for word in dictionary['word_sets'][set_name]]
            set_expressions.append('(' + ' OR '.join(predicates) + ')' if predicates else 'FALSE')
        category_expressions.append('(' + ' AND '.join(set_expressions) + ')')
    overall = '(' + ' OR '.join(category_expressions) + ')'
    aggregations = ['count(*) AS total_articles', f'sum(CAST({overall} AS BIGINT)) AS overall_gpr_articles']
    aggregations.extend(f'sum(CAST({expression} AS BIGINT)) AS category_{index}_count' for index, expression in enumerate(category_expressions))
    study = "warc_date >= TIMESTAMP '2017-01-01' AND warc_date < TIMESTAMP '2025-01-01'"
    sql = f'SELECT outlet, ({study}) AS in_study_period, ' + ', '.join(aggregations) + f' FROM {identifier(source_table)} GROUP BY outlet, in_study_period ORDER BY outlet, in_study_period'
    rows = query(name + '_category_counts_by_outlet_period', sql)
    if rows is None:
        return
    details['source_outlets'] = sorted({row['outlet'] for row in rows if row['outlet'] is not None})
    details['scope_note'] = 'all_cached_outlets means all outlets present in the named cached source table; the Lau source table is already restricted to three outlets.'
    details['category_order'] = category_names
    details['scopes'] = {}
    for scope_name, only_three, only_study in [('all_cached_outlets_all_dates', False, False), ('all_cached_outlets_2017_2024', False, True), ('three_outlets_all_dates', True, False), ('three_outlets_2017_2024', True, True)]:
        selected = [row for row in rows if (not only_three or row['outlet'] in target_outlets) and (not only_study or row['in_study_period'])]
        overall_count = sum(row['overall_gpr_articles'] or 0 for row in selected)
        category_counts = {category: sum(row[f'category_{index}_count'] or 0 for row in selected) for index, category in enumerate(category_names)}
        percentages = {category: count / overall_count * 100 if overall_count else None for category, count in category_counts.items()}
        summary = {'total_articles': sum(row['total_articles'] for row in selected), 'overall_gpr_articles': overall_count, 'category_counts': category_counts, 'category_percentage_of_overall_gpr': percentages}
        if name == 'original_president':
            summary['draft_claims_percent'] = president_claims
            summary['difference_percentage_points'] = {category: percentages[category] - claimed for category, claimed in president_claims.items()}
            summary['all_claims_match_to_two_decimals'] = all(round(percentages[category], 2) == claimed for category, claimed in president_claims.items())
        details['scopes'][scope_name] = summary

root = Path('/home5/chuenho/gpr-zero')
dictionary_audit('original_president', root / 'data/dictionaries/taiwan_gpr_dict_president.json', dictionary_dir / 'word_column_mapping.json', 'gpr_binary_matrix_all_words', all_words)
dictionary_audit('lau', root / 'data/dictionaries/yeu_tong_lau_2024.json', root / 'data/gpr_timeseries_3outlets/lau_word_column_mapping.json', 'lau_binary_matrix_3outlets')

# Reconstruct the original seven-category dictionary from existing category flags
# plus its one removed economic keyword. This reads no article text.
original = json.loads((root / 'data/dictionaries/taiwan_gpr_dict_president.json').read_text())
updated_path = root / 'data/dictionaries/taiwan_gpr_dict_president_updated.json'
updated_bytes = updated_path.read_bytes()
updated = json.loads(updated_bytes)
mapping = json.loads((dictionary_dir / 'word_column_mapping.json').read_text())
categories = list(original['categories'])
changed_sets = {key: {'original_only': sorted(set(words) - set(updated['word_sets'].get(key, []))), 'updated_only': sorted(set(updated['word_sets'].get(key, [])) - set(words))} for key, words in original['word_sets'].items() if set(words) != set(updated['word_sets'].get(key, []))}
same_categories = all(expression == updated['categories'].get(category) for category, expression in original['categories'].items())
reconstruction = {
    'labeled_table': 'gpr_labeled_taiwan_gpr_dict_president_updated',
    'updated_dictionary_path': str(updated_path),
    'updated_dictionary_sha256': hashlib.sha256(updated_bytes).hexdigest(),
    'original_category_definitions_unchanged': same_categories,
    'changed_original_word_sets': changed_sets,
    'scope': 'three_outlets_2017_2024',
    'provenance_limit': 'Uses named historical materialized flags; does not rescan article text or independently establish how those flags were created.',
}
result['original_president_labeled_reconstruction'] = reconstruction
original_mapping = result['original_president']
mapping_valid = original_mapping['mapping_values_match_binary_schema'] and original_mapping['mapping_is_one_to_one'] and original_mapping['mapping_equals_sorted_union'] and not original_mapping['missing_dictionary_words']
if mapping_valid and same_categories and changed_sets == {'economic_coercion': {'original_only': ['斷交'], 'updated_only': []}}:
    table_name = reconstruction['labeled_table']
    flags = [f'coalesce({identifier("is_" + category)}, FALSE)' for category in categories]
    overall = '(' + ' OR '.join(flags) + ')'
    aggregates = ['count(*) AS total_articles', f'sum(CAST({overall} AS BIGINT)) AS overall_gpr_articles']
    aggregates.extend(f'sum(CAST({flag} AS BIGINT)) AS category_{index}_count' for index, flag in enumerate(flags))
    study = "date >= DATE '2017-01-01' AND date < DATE '2025-01-01'"
    outlets = "outlet IN ('China Times (中國時報)', 'Liberty Times (自由時報)', 'United Daily (聯合報)')"
    weekly_sql = 'SELECT date_trunc(\'week\', date) AS week, ' + ', '.join(aggregates) + f' FROM {identifier(table_name)} WHERE {study} AND {outlets} GROUP BY week ORDER BY week'
    weekly = query('president_updated_cache_weekly_seven_categories', weekly_sql)
    correction_column = identifier(mapping['斷交'])
    correction_sql = f'''WITH candidates AS (
        SELECT warc_id, date, {overall} AS already_gpr
        FROM {identifier(table_name)}
        WHERE {study} AND {outlets}
        AND coalesce(has_cross_strait_relations, FALSE)
        AND NOT coalesce("is_Economic Coercion", FALSE)
    )
    SELECT date_trunc('week', candidates.date) AS week,
        count(*) AS additional_economic_articles,
        sum(CAST(NOT candidates.already_gpr AS BIGINT)) AS additional_overall_gpr_articles
    FROM candidates JOIN gpr_binary_matrix_all_words AS bm USING (warc_id)
    WHERE coalesce(bm.{correction_column}, FALSE)
    GROUP BY week ORDER BY week'''
    corrections = query('original_president_removed_word_correction', correction_sql) if weekly else None
    if weekly is not None and corrections is not None:
        by_week = {row['week']: row for row in corrections}
        economic_index = categories.index('Economic Coercion')
        corrected = []
        for row in weekly:
            row = dict(row)
            correction = by_week.get(row['week'], {})
            row['overall_gpr_articles'] += correction.get('additional_overall_gpr_articles', 0)
            row[f'category_{economic_index}_count'] += correction.get('additional_economic_articles', 0)
            corrected.append(row)
        total = sum(row['overall_gpr_articles'] for row in corrected)
        counts = {category: sum(row[f'category_{index}_count'] for row in corrected) for index, category in enumerate(categories)}
        percentages = {category: count / total * 100 for category, count in counts.items()}
        reconstruction.update({
            'category_order': categories,
            'total_articles': sum(row['total_articles'] for row in corrected),
            'overall_gpr_articles': total,
            'category_counts': counts,
            'category_percentage_of_overall_gpr': percentages,
            'additional_economic_articles': sum(row['additional_economic_articles'] for row in corrections),
            'additional_overall_gpr_articles': sum(row['additional_overall_gpr_articles'] for row in corrections),
            'draft_claims_percent': president_claims,
            'all_claims_match_to_two_decimals': all(round(percentages[category], 2) == claimed for category, claimed in president_claims.items()),
            'weekly_original': corrected,
        })
con.close()
print(json.dumps(result, ensure_ascii=False, default=str))
"""


def write_weekly_csv(result: dict, output: Path) -> None:
    reconstruction = result.get('original_president_labeled_reconstruction', {})
    weekly = reconstruction.get('weekly_original')
    if weekly is None:
        return
    categories = reconstruction['category_order']
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ['week', 'scope', 'total_articles', 'total_gpr_count', 'total_gpr_share']
    columns.extend(f'{category}_count' for category in categories)
    columns.extend(f'{category}_share' for category in categories)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in weekly:
            total = row['total_articles']
            exported = {'week': row['week'].split()[0], 'scope': reconstruction['scope'], 'total_articles': total, 'total_gpr_count': row['overall_gpr_articles'], 'total_gpr_share': row['overall_gpr_articles'] / total}
            exported.update({f'{category}_count': row[f'category_{index}_count'] for index, category in enumerate(categories)})
            exported.update({f'{category}_share': row[f'category_{index}_count'] / total for index, category in enumerate(categories)})
            writer.writerow(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='bdsrc')
    parser.add_argument('--remote-python', default='/home5/chuenho/gpr-zero/.venv/bin/python')
    parser.add_argument('--output', type=Path, default=Path('replication/remote_corpus_audit.json'))
    parser.add_argument('--weekly-output', type=Path, default=Path('replication/draft/weekly_original_president.csv'))
    parser.add_argument('--original-binary-counts', action='store_true', help='Opt into the broader 97-word count query; still interrupted after 45 seconds.')
    args = parser.parse_args()
    completed = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host, args.remote_python, '-'],
        input=REMOTE_CODE.replace('__RUN_ORIGINAL_BINARY_COUNTS__', repr(args.original_binary_counts)), text=True, capture_output=True, timeout=300, check=True,
    )
    result = json.loads(completed.stdout)
    result['ssh_host'] = args.host
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        previous_attempt = previous.get('queries', {}).get('original_president_category_counts_by_outlet_period') or previous.get('previous_original_binary_attempt')
        if previous_attempt and 'error' in previous_attempt:
            result['previous_original_binary_attempt'] = previous_attempt
    if completed.stderr:
        result['remote_stderr'] = completed.stderr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    write_weekly_csv(result, args.weekly_output)
    summary = {key: result[key] for key in ('python_executable', 'duckdb_version', 'database_bytes', 'metadata_count_source')}
    summary['queries'] = {key: {'seconds': value['seconds'], 'row_count': len(value.get('rows', [])), **({'error': value['error']} if 'error' in value else {})} for key, value in result['queries'].items()}
    summary['dictionary_count'] = result['dictionary_manifest']['count']
    summary['unique_word_count'] = result['dictionary_manifest']['unique_word_count']
    summary['original_president'] = result['original_president']
    summary['lau'] = result['lau']
    summary['original_president_labeled_reconstruction'] = {key: value for key, value in result['original_president_labeled_reconstruction'].items() if key != 'weekly_original'}
    summary['output'] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

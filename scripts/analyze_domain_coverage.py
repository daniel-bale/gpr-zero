#!/usr/bin/env python3
"""
Analyze Domain Coverage in JSON Files
Identifies domains with significant article counts that might be missing from target_outlets mapping
"""

import json
from pathlib import Path
import polars as pl
from collections import defaultdict
import difflib

def calculate_similarity_score(domain1: str, domain2: str) -> float:
    """Calculate similarity score between two domain names using difflib"""
    return difflib.SequenceMatcher(None, domain1.lower(), domain2.lower()).ratio()

def is_potential_taiwan_outlet(domain: str, article_count: int) -> tuple[bool, str]:
    """Check if domain could be a Taiwan news outlet based on patterns"""
    domain_lower = domain.lower()
    
    # Minimum threshold for consideration
    if article_count < 500:
        return False, "Too few articles"
    
    # Taiwan-specific patterns
    taiwan_indicators = [
        'taiwan', 'taipei', 'formosa', 'roc', 'republic',
        'chinese', 'china', 'asian', 'orient'
    ]
    
    # News-related patterns  
    news_indicators = [
        'news', 'daily', 'times', 'post', 'herald', 'tribune',
        'journal', 'gazette', 'press', 'media', 'radio', 'tv',
        'broadcast', 'report', 'today', 'weekly', 'magazine'
    ]
    
    # Check for Taiwan indicators
    has_taiwan_indicator = any(indicator in domain_lower for indicator in taiwan_indicators)
    
    # Check for news indicators
    has_news_indicator = any(indicator in domain_lower for indicator in news_indicators)
    
    # Special cases for major international outlets that might cover Taiwan
    major_outlets = [
        'reuters', 'bloomberg', 'wsj', 'nytimes', 'washingtonpost',
        'bbc', 'cnn', 'foxnews', 'nbcnews', 'cbsnews', 'abcnews',
        'guardian', 'economist', 'ft.com', 'ap.org', 'afp',
        'scmp', 'straitstimes', 'japantimes', 'koreaherald'
    ]
    
    # Check if it's a major international outlet
    is_major_international = any(outlet in domain_lower for outlet in major_outlets)
    
    # Scoring logic
    if has_taiwan_indicator and has_news_indicator:
        return True, "Taiwan + News indicators"
    elif has_taiwan_indicator and article_count > 2000:
        return True, "Taiwan indicator + high volume"
    elif is_major_international and article_count > 5000:
        return True, "Major international outlet"
    elif has_news_indicator and article_count > 10000:
        return True, "News outlet + very high volume"
    elif 'chinese' in domain_lower and has_news_indicator:
        return True, "Chinese + News indicators"
    
    return False, "No clear Taiwan/news patterns"

def find_similar_domains(target_domains: set[str], untracked_domains: list[dict], threshold: float = 0.6) -> list[dict]:
    """Find untracked domains similar to existing tracked domains"""
    similar_domains = []
    
    for untracked in untracked_domains:
        untracked_domain = untracked['domain']
        untracked_base = untracked_domain.split('.')[0].lower()
        
        for tracked_domain in target_domains:
            tracked_base = tracked_domain.split('.')[0].lower()
            
            # Calculate similarity score
            similarity = calculate_similarity_score(untracked_base, tracked_base)
            
            if similarity >= threshold:
                similar_domains.append({
                    'untracked_domain': untracked_domain,
                    'articles': untracked['total_articles'],
                    'similar_to': tracked_domain,
                    'similarity_score': similarity
                })
    
    return sorted(similar_domains, key=lambda x: (x['similarity_score'], x['articles']), reverse=True)

def analyze_domain_coverage():
    """Analyze all domains in JSON domain count files to identify potentially missed outlets"""
    
    print("="*80)
    print("ENHANCED DOMAIN COVERAGE ANALYSIS")
    print("Scanning JSON domain count files for potentially missed Taiwan news outlets")
    print("="*80)
    
    # JSON domain count files
    json_files = [
        "logs/warc_extract/stage1/stage1_domain_counts_20250509_231321.json",
        "logs/warc_extract/stage1/stage1_domain_counts_20250512_031450.json", 
        "logs/warc_extract/stage1/stage1_domain_counts_20250513_024646.json",
        "logs/warc_extract/stage1/stage1_domain_counts_20250514_003739.json",
        "logs/warc_extract/stage1/stage1_domain_counts_20250514_175031.json",
        "logs/warc_extract/stage1/stage1_domain_counts_20250515_102606.json",
        "logs/warc_extract/stage1/stage1_domain_counts_20250516_000823.json",
        "logs/warc_extract/stage1/stage1_domain_counts_20250516_005129.json",
        "logs/warc_extract/stage1/stage1_domain_counts_20250528_013917.json",
    ]
    
    # Current target outlets we're tracking
    target_outlets = {
        'cna.com.tw': 'Central News Agency (中央社)',
        'chinatimes.com': 'China Times (中國時報)',
        'news.ltn.com.tw': 'Liberty Times (自由時報)', 
        'udn.com': 'United Daily (聯合報)',
        'peoplenews.tw': 'People News (民報)',
        'ctee.com.tw': 'Commercial Times (工商時報)',
        'ettoday.net': 'ETtoday (ETtoday新聞雲)',
        'thenewslens.com': 'The News Lens (關鍵評論網)',
        'taiwannews.com.tw': 'Taiwan News',
        'nownews.com': 'Now News (今日新聞)'
    }
    
    print(f"Currently tracking {len(target_outlets)} outlets:")
    for domain, name in target_outlets.items():
        print(f"  • {domain} → {name}")
    print()
    
    # Aggregate domain counts across all JSON files
    all_domain_counts = defaultdict(int)
    file_count = 0
    
    for json_file in json_files:
        json_path = Path(json_file)
        if json_path.exists():
            file_count += 1
            print(f"Processing {json_file}...")
            
            with open(json_path, 'r') as f:
                domain_counts = json.load(f)
            
            for domain, count in domain_counts.items():
                all_domain_counts[domain] += count
        else:
            print(f"⚠️  JSON file not found: {json_path}")
    
    print(f"\nProcessed {file_count} JSON files")
    print(f"Found {len(all_domain_counts)} unique domains total")
    
    # Convert to DataFrame for analysis
    domain_data = [
        {
            'domain': domain,
            'total_articles': count,
            'is_tracked': domain in target_outlets,
            'outlet_name': target_outlets.get(domain, 'Unknown')
        }
        for domain, count in all_domain_counts.items()
    ]
    
    df = pl.DataFrame(domain_data).sort('total_articles', descending=True)
    
    # Analysis 1: Top domains by article count
    print("\n" + "="*80)
    print("TOP 20 DOMAINS BY TOTAL ARTICLE COUNT")
    print("="*80)
    print(f"{'Rank':<4} {'Domain':<30} {'Articles':<12} {'Tracked':<8} {'Outlet Name'}")
    print("-" * 80)
    
    top_domains = df.head(20)
    for i, row in enumerate(top_domains.iter_rows(named=True), 1):
        tracked_status = "✅ YES" if row['is_tracked'] else "❌ NO"
        outlet_display = row['outlet_name'] if row['is_tracked'] else "NOT TRACKED"
        print(f"{i:<4} {row['domain']:<30} {row['total_articles']:>11,} {tracked_status:<8} {outlet_display}")
    
    # Analysis 2: Enhanced analysis of significant untracked domains
    print("\n" + "="*80)  
    print("POTENTIALLY MISSED TAIWAN NEWS OUTLETS")
    print("Enhanced analysis of untracked domains with >500 total articles")
    print("="*80)
    
    untracked_significant = df.filter(
        (~pl.col('is_tracked')) & (pl.col('total_articles') > 500)
    ).sort('total_articles', descending=True)
    
    if untracked_significant.is_empty():
        print("✅ No untracked domains found with >500 articles")
    else:
        potential_taiwan_outlets = []
        
        print(f"{'Domain':<35} {'Articles':<10} {'Potential Taiwan Outlet':<8} {'Reason'}")
        print("-" * 85)
        
        for row in untracked_significant.iter_rows(named=True):
            domain = row['domain']
            articles = row['total_articles']
            
            is_potential, reason = is_potential_taiwan_outlet(domain, articles)
            potential_status = "🎯 YES" if is_potential else "❌ NO"
            
            print(f"{domain:<35} {articles:>9,} {potential_status:<8} {reason}")
            
            if is_potential:
                potential_taiwan_outlets.append({
                    'domain': domain,
                    'articles': articles,
                    'reason': reason
                })
        
        print(f"\n🎯 Found {len(potential_taiwan_outlets)} potential Taiwan news outlets to investigate!")
    
    # Analysis 3: String similarity analysis
    print("\n" + "="*80)
    print("STRING SIMILARITY ANALYSIS")
    print("Finding untracked domains similar to existing tracked outlets")
    print("="*80)
    
    untracked_domains_list = [
        {'domain': row['domain'], 'total_articles': row['total_articles']}
        for row in df.filter(~pl.col('is_tracked')).iter_rows(named=True)
        if row['total_articles'] > 100  # Only check domains with some significance
    ]
    
    similar_domains = find_similar_domains(set(target_outlets.keys()), untracked_domains_list)
    
    if similar_domains:
        print(f"{'Untracked Domain':<30} {'Articles':<10} {'Similar To':<25} {'Score':<6} {'Tracked Outlet'}")
        print("-" * 95)
        for sim in similar_domains[:20]:  # Show top 20
            tracked_outlet = target_outlets[sim['similar_to']]
            print(f"{sim['untracked_domain']:<30} {sim['articles']:>9,} {sim['similar_to']:<25} {sim['similarity_score']:.3f} {tracked_outlet}")
    else:
        print("✅ No similar domains found with similarity score ≥ 0.6")
    
    # Analysis 4: Coverage statistics
    print("\n" + "="*80)
    print("COVERAGE STATISTICS")
    print("="*80)
    
    total_articles_all = df['total_articles'].sum()
    tracked_articles = df.filter(pl.col('is_tracked'))['total_articles'].sum()
    coverage_percentage = (tracked_articles / total_articles_all) * 100 if total_articles_all > 0 else 0
    
    print(f"Total articles across all domains: {total_articles_all:,}")
    print(f"Articles from tracked outlets: {tracked_articles:,}")
    print(f"Coverage percentage: {coverage_percentage:.2f}%")
    print(f"Tracked domains: {df.filter(pl.col('is_tracked')).height}")
    print(f"Untracked domains: {df.filter(~pl.col('is_tracked')).height}")
    
    # Analysis 5: Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if 'potential_taiwan_outlets' in locals() and potential_taiwan_outlets:
        print("🎯 POTENTIAL TAIWAN OUTLETS TO INVESTIGATE:")
        for outlet in sorted(potential_taiwan_outlets, key=lambda x: x['articles'], reverse=True)[:10]:
            print(f"   • {outlet['domain']} ({outlet['articles']:,} articles) - {outlet['reason']}")
        print()
    
    if similar_domains:
        print("🔄 SIMILAR DOMAINS TO EXISTING OUTLETS:")
        for sim in similar_domains[:5]:
            print(f"   • {sim['untracked_domain']} → similar to {sim['similar_to']} (score: {sim['similarity_score']:.3f})")
        print()
    
    print("📋 NEXT STEPS:")
    print("   1. Manually investigate the potential Taiwan outlets listed above")
    print("   2. Check if similar domains are alternative URLs for existing outlets")
    print("   3. Consider adding legitimate Taiwan news sources to target_outlets mapping")
    
    # Save results
    output_dir = Path("docs/debugging/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_domains_path = output_dir / "all_domains_analysis_enhanced.csv"
    df.write_csv(all_domains_path)
    print(f"\n📁 Full domain analysis saved to: {all_domains_path}")
    
    if 'potential_taiwan_outlets' in locals() and potential_taiwan_outlets:
        potential_path = output_dir / "potential_taiwan_outlets.csv"
        potential_df = pl.DataFrame(potential_taiwan_outlets)
        potential_df.write_csv(potential_path)
        print(f"📁 Potential Taiwan outlets saved to: {potential_path}")
    
    if similar_domains:
        similar_path = output_dir / "similar_domains_analysis.csv"
        similar_df = pl.DataFrame(similar_domains)
        similar_df.write_csv(similar_path)
        print(f"📁 Similar domains analysis saved to: {similar_path}")
    
    print("\n" + "="*80)
    print("✅ ENHANCED DOMAIN COVERAGE ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    analyze_domain_coverage() 
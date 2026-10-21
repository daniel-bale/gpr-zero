import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_thesis_profile.py"
spec = importlib.util.spec_from_file_location("export_thesis_profile", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileGuardTests(unittest.TestCase):
    def test_producer_object_hash_uses_the_aggregator_encoding(self):
        value = {"word": "軍演", "mode": "full"}
        expected = profile.digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())
        self.assertEqual(profile.object_hash(value), expected)
        self.assertNotEqual(profile.object_hash(value), profile.digest(profile.json_bytes(value)))

    def test_incomplete_diagnostic_and_utc_inputs_never_promoted(self):
        complete = dict(status="complete", source_mode="full", complete=True,
                        input_complete=True, source_complete=True, calendar_timezone="Asia/Taipei")
        profile.complete_aggregate(complete)
        for key, value in [("status", "running"), ("source_mode", "benchmark"), ("complete", False),
                           ("input_complete", False), ("source_complete", False), ("calendar_timezone", "UTC")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                profile.complete_aggregate({**complete, key: value})
        with self.assertRaises(ValueError):
            profile.complete_aggregate({key: value for key, value in complete.items() if key != "source_complete"})

    def test_pairwise_selects_all_outlets_study_labels_and_requires_full_panel(self):
        rows = []
        for step, dictionaries, value in (("step1", 35, .493), ("step2", 40, .698)):
            metrics = dict(dictionary_count=dictionaries, weeks=418, pair_count=dictionaries*(dictionaries-1)//2,
                           incomplete_weeks=0, mean_pairwise_correlation=value)
            rows.append(dict(step=step, scope="all_outlets", period="strict_study_week_labels", reference=metrics, rebuilt=metrics.copy()))
            rows.append(dict(step=step, scope="three_outlets", period="strict_study_week_labels", reference={}, rebuilt={}))
            rows.append(dict(step=step, scope="all_outlets", period="full_cache", reference={}, rebuilt={}))
        selected = profile.select_pairwise(rows)
        self.assertEqual([row["mean_pairwise_correlation"] for row in selected], [.493, .698])
        broken = copy.deepcopy(rows)
        broken[0]["rebuilt"]["incomplete_weeks"] = 1
        with self.assertRaises(ValueError):
            profile.select_pairwise(broken)
        with self.assertRaises(ValueError):
            profile.select_pairwise(rows + [rows[0]])

    def test_variant_swaps_and_missing_or_duplicate_rows_fail(self):
        rows = [{"week": "2024-12-30", "category": "Military", "category_share": ".1"}]
        evidence = [{**rows[0], "category_share_reference": ".1", "category_share_rebuilt": ".1",
                     "coverage": "both", "numeric_mismatch": "False"}]
        profile.validate_comparison(rows, evidence, ["week", "category"], ["category_share"])
        for wrong in ([], rows*2, [{**rows[0], "category_share": ".11"}]):
            with self.assertRaises(ValueError):
                profile.validate_comparison(wrong, evidence, ["week", "category"], ["category_share"])
        with self.assertRaises(ValueError):
            profile.validate_comparison(rows, [{**evidence[0], "coverage": "right_only"}], ["week", "category"], ["category_share"])

    def test_weekly_projection_selects_rebuilt_values_and_requires_equal_reference(self):
        rows = [{"week": "2020-01-06", "gpr_share_reference": ".01", "gpr_share_rebuilt": ".01",
                 "coverage": "both", "numeric_mismatch": "False"}]
        self.assertEqual(profile.project_comparison(rows, ["week"], ["gpr_share"]),
                         [{"week": "2020-01-06", "gpr_share": ".01"}])
        with self.assertRaises(ValueError):
            profile.project_comparison([{**rows[0], "gpr_share_reference": ".02"}], ["week"], ["gpr_share"])

    def test_var_projects_only_historical_scaling_and_checks_every_response(self):
        rows = [dict(version="model", period=str(period), response=f"v{variable}", impulse="log_GPR",
                     legacy_refit=".2", historical_cached=".2", legacy_display_multiplier="100",
                     legacy_display_unit="100 log points", two_sd_consistent_display="99")
                for variable in range(7) for period in range(21)]
        result = profile.legacy_var_rows(rows, "model")
        self.assertEqual(len(result), 147)
        self.assertEqual(result[0]["point_estimate"], ".2")
        self.assertNotIn("two_sd_consistent_display", result[0])
        for wrong in (rows[:-1], rows[:-1]+[rows[0]], [{**rows[0], "legacy_refit": "99"}]+rows[1:]):
            with self.assertRaises(ValueError):
                profile.legacy_var_rows(wrong, "model")

    def test_current_thesis_labels_use_corrected_source_not_old_claims(self):
        dates = [f"2024-01-{day:02}" for day in range(1, 12)]
        text = "label={tab:unique-spikes}\n" + "\n".join(f"{day} & Event" for day in dates) + "\n\\bottomrule\n"
        text += "between China Times and Liberty Times is 0.559, between China Times and United Daily is 0.78 and between Liberty Times and United Daily is 0.475."
        correlations = [{"": "China Times", "Liberty Times": ".559129", "United Daily": ".779783"},
                        {"": "Liberty Times", "United Daily": ".474692"}]
        result = profile.corrected_target_checks(text, [{"peak_date": day} for day in dates], correlations)
        self.assertTrue(result["unique_peak_dates_match"])
        for wrong in (text.replace("0.559", "0.475"), text.replace(dates[0], "2024-11-04")):
            with self.assertRaises(ValueError):
                profile.corrected_target_checks(wrong, [{"peak_date": day} for day in dates], correlations)

    def test_category_figure_uses_covered_current_events_not_word_table_ranges(self):
        events = [dict(start_date="2024-12-23", end_date="2024-12-30", peak_date="2024-12-30"),
                  dict(start_date="2025-03-03", end_date="2025-03-03", peak_date="2025-03-03")]
        categories = [dict(week=week, category=f"category{category}", category_share=str(value))
                      for week, value in [("2024-12-23", 1), ("2024-12-30", 3)] for category in range(8)]
        overall = [dict(week="2024-12-23", median_raw="2"), dict(week="2024-12-30", median_raw="6"),
                   dict(week="2025-03-03", median_raw="9")]
        ratios, ranges = profile.category_spike_rows(events, categories, overall)
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0]["spike_id"], "S01")
        self.assertEqual(len(ratios), 8)
        self.assertTrue(all(row["ratio"] == .5 for row in ratios))

    def test_stale_hash_and_unbound_manifest_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text('{"complete":true}')
            exporter = profile.Export(root, root)
            evidence = {"path": str(path), "sha256": profile.digest(path.read_bytes())}
            exporter.bound([evidence], path)
            with self.assertRaises(ValueError):
                exporter.bound([], path)
            with self.assertRaises(ValueError):
                exporter.bound([evidence, evidence], path)
            path.write_text('{"complete":false}')
            with self.assertRaises(ValueError):
                exporter.bound([evidence], path)

    def test_publish_hashes_outputs_and_refuses_input_mutation_or_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("a\n1\n")
            exporter = profile.Export(root, root)
            exporter.copy_csv(source, "canonical.csv", "historical", 1)
            manifest = {"profile": profile.PROFILE, "artifacts": exporter.artifacts}
            output = root / "profile"
            exporter.publish(output, manifest)
            self.assertEqual((output / "canonical.csv").read_bytes(), source.read_bytes())
            recorded = json.loads((output / "manifest.json").read_text())
            self.assertEqual(recorded["artifacts"]["canonical.csv"]["sha256"], profile.digest(source.read_bytes()))
            exporter.publish(output, manifest, overwrite=True)
            (output / "user-notes.txt").write_text("preserve")
            with self.assertRaises(ValueError):
                exporter.publish(output, manifest, overwrite=True)
            source.write_text("a\n2\n")
            with self.assertRaises(ValueError):
                exporter.publish(root / "other", manifest)
            self.assertFalse((root / "other").exists())

    def test_publish_rejects_symlink_before_resolving_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            exporter = profile.Export(root, root)
            with self.assertRaisesRegex(ValueError, "symlink"):
                exporter.publish(link, {"profile": profile.PROFILE, "artifacts": {}})
            self.assertTrue(link.is_symlink())
            self.assertEqual(list(real.iterdir()), [])

    def test_overwrite_preserves_edited_artifacts_and_same_named_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("a\n1\n")
            exporter = profile.Export(root, root)
            exporter.copy_csv(source, "canonical.csv", "historical", 1)
            manifest = {"profile": profile.PROFILE, "artifacts": exporter.artifacts}
            output = root / "profile"
            exporter.publish(output, manifest)
            artifact = output / "canonical.csv"
            artifact.write_text("a\n2\n")
            with self.assertRaises(ValueError):
                exporter.publish(output, manifest, overwrite=True)
            self.assertEqual(artifact.read_text(), "a\n2\n")
            artifact.unlink()
            artifact.mkdir()
            (artifact / "user.txt").write_text("preserve")
            with self.assertRaises(ValueError):
                exporter.publish(output, manifest, overwrite=True)
            self.assertEqual((artifact / "user.txt").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()

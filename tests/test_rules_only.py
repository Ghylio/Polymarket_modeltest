import unittest

from research.rules_only import extract_rules_features


class RulesOnlyTests(unittest.TestCase):
    def test_ambiguous_when_rules_missing(self):
        features = extract_rules_features({"question": "Will it rain?"})
        self.assertEqual(features["rules_ambiguous"], 1)
        self.assertGreaterEqual(features["ambiguity_score"], 0.5)

    def test_objective_when_official_published(self):
        features = extract_rules_features(
            {
                "rules": "Outcome based on official published government data",
            }
        )
        self.assertEqual(features["rules_ambiguous"], 0)
        self.assertLess(features["ambiguity_score"], 0.5)
        self.assertIn(features["resolution_source_type"], {"official_statement", "government_data"})


if __name__ == "__main__":
    unittest.main()

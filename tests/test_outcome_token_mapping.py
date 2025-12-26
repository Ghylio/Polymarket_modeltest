import unittest

from polymarket_fetcher import (
    PolymarketFetcher,
    build_outcome_token_map,
)


class TestOutcomeTokenMapping(unittest.TestCase):
    def setUp(self):
        self.fetcher = PolymarketFetcher()

    def test_yes_no_order(self):
        market = {
            'id': '1',
            'outcomes': ['Yes', 'No'],
            'clobTokenIds': ['A', 'B'],
        }
        mapping = build_outcome_token_map(market)
        self.assertEqual(mapping['yes_token_id'], 'A')
        self.assertEqual(mapping['no_token_id'], 'B')
        yes_token, no_token = self.fetcher.get_token_ids_for_market(market)
        self.assertEqual(yes_token, 'A')
        self.assertEqual(no_token, 'B')

    def test_no_yes_order(self):
        market = {
            'id': '2',
            'outcomes': ['No', 'Yes'],
            'clobTokenIds': ['A', 'B'],
        }
        mapping = build_outcome_token_map(market)
        self.assertEqual(mapping['yes_token_id'], 'B')
        self.assertEqual(mapping['no_token_id'], 'A')
        yes_token, no_token = self.fetcher.get_token_ids_for_market(market)
        self.assertEqual(yes_token, 'B')
        self.assertEqual(no_token, 'A')

    def test_multi_outcome_mapping(self):
        market = {
            'id': '3',
            'outcomes': ['A', 'B', 'C'],
            'clobTokenIds': ['1', '2', '3'],
        }
        mapping = build_outcome_token_map(market)
        self.assertEqual(len(mapping['outcome_token_map']), 3)
        yes_token, no_token = self.fetcher.get_token_ids_for_market(market)
        self.assertEqual(yes_token, '1')
        self.assertEqual(no_token, '2')

    def test_json_string_lists(self):
        market = {
            'id': '4',
            'outcomes': '["Yes", "No"]',
            'clobTokenIds': '["token1", "token2"]',
        }
        mapping = build_outcome_token_map(market)
        self.assertEqual(mapping['yes_token_id'], 'token1')
        self.assertEqual(mapping['no_token_id'], 'token2')
        yes_token, no_token = self.fetcher.get_token_ids_for_market(market)
        self.assertEqual(yes_token, 'token1')
        self.assertEqual(no_token, 'token2')

    def test_length_mismatch(self):
        market = {
            'id': '5',
            'outcomes': ['Yes'],
            'clobTokenIds': ['token1', 'token2'],
        }
        with self.assertRaises(ValueError):
            build_outcome_token_map(market)


if __name__ == '__main__':
    unittest.main()

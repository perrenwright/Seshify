import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure local directories can be imported
sys.path.append(".")

from nodes import extract_metric, judge_conditions, send_notification, parse_coordinate
from tools import fetch_stormglass_data, send_telegram_message

class TestSurfAgent(unittest.TestCase):
    
    def test_parse_coordinate(self):
        # Case 1: Empty string (like missing GitHub Actions secret)
        self.assertEqual(parse_coordinate("", 33.6839), 33.6839)
        # Case 2: Whitespace
        self.assertEqual(parse_coordinate("   ", -118.0122), -118.0122)
        # Case 3: None
        self.assertEqual(parse_coordinate(None, 33.6839), 33.6839)
        # Case 4: Valid numeric string
        self.assertEqual(parse_coordinate("34.0259", 33.6839), 34.0259)
        # Case 5: Invalid string
        self.assertEqual(parse_coordinate("not-a-number", 33.6839), 33.6839)
    
    def test_extract_metric(self):
        # Case 1: Simple dict
        hour_data = {"waveHeight": {"sg": 1.5, "noaa": 1.8}}
        self.assertEqual(extract_metric(hour_data, "waveHeight"), 1.5)
        
        # Case 2: Fallback when 'sg' is missing
        hour_data_no_sg = {"waveHeight": {"noaa": 1.8}}
        self.assertEqual(extract_metric(hour_data_no_sg, "waveHeight"), 1.8)
        
        # Case 3: Empty dictionary
        hour_data_empty = {"waveHeight": {}}
        self.assertEqual(extract_metric(hour_data_empty, "waveHeight"), 0.0)
        
        # Case 4: Non-dict values
        hour_data_flat = {"waveHeight": 2.2}
        self.assertEqual(extract_metric(hour_data_flat, "waveHeight"), 2.2)

    def test_hard_filter_stay_wave_period(self):
        # wavePeriod < 5 -> STAY
        state = {
            "forecast_data": {
                "hours": [{
                    "wavePeriod": {"sg": 4.0},
                    "windSpeed": {"sg": 3.0},
                    "waveHeight": {"sg": 1.5},
                    "windDirection": {"sg": 180.0}
                }]
            }
        }
        res = judge_conditions(state)
        self.assertEqual(res["decision"], "STAY")
        self.assertIn("Wave period is too short", res["reasoning"])

    def test_hard_filter_stay_wind_speed(self):
        # windSpeed > 10 -> STAY
        state = {
            "forecast_data": {
                "hours": [{
                    "wavePeriod": {"sg": 8.0},
                    "windSpeed": {"sg": 12.0},
                    "waveHeight": {"sg": 1.5},
                    "windDirection": {"sg": 180.0}
                }]
            }
        }
        res = judge_conditions(state)
        self.assertEqual(res["decision"], "STAY")
        self.assertIn("wind speed is too strong", res["reasoning"])

    @patch("nodes.get_llm")
    def test_llm_reasoner_go(self, mock_get_llm):
        # Set up a fake LLM response
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        from nodes import SurfDecision
        mock_structured_llm.invoke.return_value = SurfDecision(
            decision="GO",
            reasoning="The conditions are perfect with a clean swell!"
        )
        
        state = {
            "forecast_data": {
                "hours": [{
                    "wavePeriod": {"sg": 8.0},
                    "windSpeed": {"sg": 4.0},
                    "waveHeight": {"sg": 2.0},
                    "windDirection": {"sg": 180.0}
                }]
            }
        }
        
        res = judge_conditions(state)
        self.assertEqual(res["decision"], "GO")
        self.assertEqual(res["reasoning"], "The conditions are perfect with a clean swell!")

    @patch("nodes.send_telegram_message")
    @patch.dict(os.environ, {"TG_TOKEN": "test_token", "TG_CHAT_ID": "test_chat"})
    def test_send_notification_go(self, mock_send):
        state = {
            "decision": "GO",
            "reasoning": "Swell looks perfect today!"
        }
        send_notification(state)
        mock_send.assert_called_once_with(
            "test_token", 
            "test_chat", 
            "🏄‍♂️ <b>SURF SENTRY CHECK: GO!</b> 🏄‍♂️\n\nSwell looks perfect today!\n\n<i>Have a legendary session! 🌊</i>"
        )

    @patch("nodes.send_telegram_message")
    def test_send_notification_stay(self, mock_send):
        state = {
            "decision": "STAY",
            "reasoning": "Conditions are poor."
        }
        send_notification(state)
        mock_send.assert_not_called()

if __name__ == "__main__":
    unittest.main()

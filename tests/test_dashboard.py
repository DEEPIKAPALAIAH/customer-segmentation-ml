import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardUploadTests(unittest.TestCase):
    def test_three_customer_upload_trains_without_invalid_slider_range(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        app.sidebar.radio[0].set_value("Upload a CSV").run()
        csv_data = (
            "Age,Annual Income (k$),Spending Score (1-100)\n"
            "20,30,40\n"
            "35,50,60\n"
            "50,70,80\n"
        ).encode("utf-8")
        app.file_uploader[0].upload("three_customers.csv", csv_data, "text/csv").run()

        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertEqual(len(app.sidebar.slider), 0)
        self.assertTrue(
            any("Using 2 clusters" in item.value for item in app.sidebar.caption)
        )


if __name__ == "__main__":
    unittest.main()

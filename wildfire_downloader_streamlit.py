import pandas as pd
import streamlit as st
from wildfire.ui.app_controller import StreamlitWorkflowApp
from wildfire.ui.common import discover_feature_inputs as _discover_feature_inputs
from wildfire.ui.common import download_id as _download_id
from wildfire.ui.common import existing_matching_file as _existing_matching_file
from wildfire.ui.common import write_manifest as _write_manifest
from wildfire.ui.downloads import download_firms_area as _download_firms_area
from wildfire.ui.downloads import search_download_earthaccess as _search_download_earthaccess
from wildfire.ui.map_viewer import prepare_firms_map_points as _prepare_firms_map_points
from wildfire.ui.workflows import WildfireWorkflowPages


def main():
    pages = WildfireWorkflowPages()
    StreamlitWorkflowApp(
        title="Wildfire Earthdata Downloader",
        caption="Download FIRMS, GRACE, ERA5, and MCD64A1 data for wildfire ML backtesting.",
        workflows={
            "FIRMS active fire CSV": pages.render_firms,
            "MCD14DL real-time fire growth": pages.render_mcd14dl,
            "GRACE groundwater/root-zone moisture": pages.render_grace,
            "ERA5 temperature/dewpoint": pages.render_era5,
            "MCD64A1 burned area": pages.render_mcd64a1,
            "Map viewer": pages.render_map_viewer,
            "Build feature table": pages.render_build_features,
            "Run backtest": pages.render_backtest,
        },
    ).run()


if __name__ == "__main__":
    main()

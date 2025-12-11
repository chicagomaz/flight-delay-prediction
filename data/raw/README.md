# Data Directory

## Flight Data

The raw flight data file (`dot_flights_5years.csv`, 3.2GB) is excluded from the repository due to GitHub's file size limits.

**To obtain the data:**
1. Download 60 monthly CSV files from [Bureau of Transportation Statistics](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK)
2. Select: **Reporting Carrier On-Time Performance** (2020-2024)
3. Use the `scripts/extract_and_combine_bts.sh` script to combine all files

**Data Statistics:**
- **Total Records**: 31,339,856 flights
- **Time Period**: January 2020 - December 2024
- **File Size**: 3.2GB CSV
- **Fields**: 109 columns including departure/arrival times, delays, carrier info, airports, etc.

**Or download from:** [Alternative source if available]

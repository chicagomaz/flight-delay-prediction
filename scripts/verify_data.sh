#!/bin/bash
# Verify downloaded flight data

echo "========================================="
echo "Flight Data Verification"
echo "========================================="
echo ""

# Check if data directory exists
if [ ! -d "data/raw" ]; then
    echo "❌ Error: data/raw directory not found"
    echo "   Creating directory..."
    mkdir -p data/raw
    echo "   ✅ Created data/raw/"
    echo ""
fi

# Look for any CSV files in data/raw
CSV_FILES=$(find data/raw -name "*.csv" -type f)

if [ -z "$CSV_FILES" ]; then
    echo "❌ No CSV files found in data/raw/"
    echo ""
    echo "📥 Please download data first:"
    echo "   1. Visit: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ"
    echo "   2. Select required fields (see scripts/download_instructions.md)"
    echo "   3. Download and extract to data/raw/"
    echo ""
    echo "   OR use sample data:"
    echo "   Visit: https://www.kaggle.com/datasets/usdot/flight-delays"
    echo ""
    exit 1
fi

# Check each CSV file
for FILE in $CSV_FILES; do
    echo "📄 Found: $FILE"

    # Get file size
    SIZE=$(du -h "$FILE" | cut -f1)
    echo "   Size: $SIZE"

    # Count lines (records)
    LINES=$(wc -l < "$FILE")
    RECORDS=$((LINES - 1))  # Subtract header
    echo "   Records: $(printf "%'d" $RECORDS)"

    # Check header
    echo ""
    echo "   Header columns:"
    head -1 "$FILE" | tr ',' '\n' | nl

    # Check for required columns
    HEADER=$(head -1 "$FILE")

    REQUIRED_COLS=(
        "CARRIER\|OP_CARRIER"
        "ORIGIN"
        "DEST"
        "ARR_DELAY"
        "MONTH"
    )

    echo ""
    echo "   Checking required columns:"
    ALL_PRESENT=true

    for COL in "${REQUIRED_COLS[@]}"; do
        if echo "$HEADER" | grep -qi "$COL"; then
            echo "   ✅ $COL"
        else
            echo "   ❌ $COL (missing)"
            ALL_PRESENT=false
        fi
    done

    echo ""

    # Show first few data rows
    echo "   First 3 data rows:"
    head -4 "$FILE" | tail -3

    echo ""
    echo "========================================="

    if [ "$ALL_PRESENT" = true ]; then
        echo "✅ Data file is valid and ready to process!"
        echo ""
        echo "📊 File statistics:"
        echo "   Location: $FILE"
        echo "   Size: $SIZE"
        echo "   Records: $(printf "%'d" $RECORDS)"
        echo ""
        echo "🚀 Next steps:"
        echo "   1. To test with sample: ./scripts/quick_test.sh"
        echo "   2. To run full pipeline: ./scripts/auto_train_after_download.sh"
        echo ""
    else
        echo "⚠️  Warning: Some required columns are missing"
        echo "   The pipeline may still work if column names are different"
        echo "   Check the data schema in the output above"
        echo ""
    fi
done

exit 0

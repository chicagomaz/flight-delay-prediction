#!/bin/bash
# Extract and combine BTS flight data from multiple ZIP files

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     BTS Flight Data - Extract and Combine                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
DOWNLOAD_DIR="${1:-/mnt/g/Downloads}"
PROJECT_DIR="/mnt/c/Users/chica/Projects/PERSONALPROJECTS/flight-delay-predictionv2"
WORK_DIR="$PROJECT_DIR/data/raw/temp_extract"
OUTPUT_FILE="$PROJECT_DIR/data/raw/dot_flights_5years.csv"

echo "📂 Looking for ZIP files in: $DOWNLOAD_DIR"
echo ""

# Create working directory
mkdir -p "$WORK_DIR"

# Count ZIP files
ZIP_COUNT=$(find "$DOWNLOAD_DIR" -name "T_ONTIME_REPORTING*.zip" -type f 2>/dev/null | wc -l)

if [ $ZIP_COUNT -eq 0 ]; then
    echo "❌ No ZIP files found!"
    echo ""
    echo "Expected files like:"
    echo "  T_ONTIME_REPORTING_20251210_151919.zip"
    echo ""
    echo "Current files in Downloads:"
    ls -lh "$DOWNLOAD_DIR" | grep -i "ontime" | head -20
    exit 1
fi

echo "✅ Found $ZIP_COUNT ZIP files"
echo ""

# Start fresh
rm -f "$OUTPUT_FILE"

first_file=true
file_count=0

# Extract and combine files one at a time (sorted by timestamp in filename)
echo "📦 Processing ZIP files..."
echo ""

for zipfile in $(find "$DOWNLOAD_DIR" -name "T_ONTIME_REPORTING*.zip" -type f | sort); do
    if [ -f "$zipfile" ]; then
        ((file_count++))
        filename=$(basename "$zipfile")

        # Extract to temp directory
        echo "  [$file_count/$ZIP_COUNT] Processing: $filename"
        unzip -q -o "$zipfile" -d "$WORK_DIR" 2>/dev/null

        # Check if extraction succeeded
        if [ ! -f "$WORK_DIR/T_ONTIME_REPORTING.csv" ]; then
            echo "    ⚠️  Warning: No CSV found in $filename"
            continue
        fi

        # Append to combined file
        if [ "$first_file" = true ]; then
            # Keep header from first file
            echo "    ✅ Including header + data"
            cat "$WORK_DIR/T_ONTIME_REPORTING.csv" > "$OUTPUT_FILE"
            first_file=false
        else
            # Skip header (line 1) for subsequent files
            echo "    ✅ Adding data (skipping header)"
            tail -n +2 "$WORK_DIR/T_ONTIME_REPORTING.csv" >> "$OUTPUT_FILE"
        fi

        # Clean up extracted CSV to avoid confusion
        rm -f "$WORK_DIR/T_ONTIME_REPORTING.csv"
    fi
done

echo ""
echo "✅ Combined $file_count CSV files"
echo ""

# Get file size and line count
if [ -f "$OUTPUT_FILE" ]; then
    file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
    line_count=$(wc -l < "$OUTPUT_FILE")
    record_count=$((line_count - 1))  # Subtract header

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    SUCCESS!                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Combined Dataset:"
    echo "   Location: $OUTPUT_FILE"
    echo "   Size: $file_size"
    echo "   Records: $(printf "%'d" $record_count)"
    echo ""

    # Show first few lines
    echo "📋 First 5 lines (with line numbers):"
    head -5 "$OUTPUT_FILE" | nl
    echo ""

    # Clean up temp directory
    read -p "🗑️  Remove temporary directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$WORK_DIR"
        echo "✅ Cleaned up temporary files"
    else
        echo "ℹ️  Temporary files kept in: $WORK_DIR"
    fi

    echo ""
    echo "🎯 Next Steps:"
    echo "   1. Verify: ./scripts/verify_data.sh"
    echo "   2. Train:  ./scripts/auto_train_after_download.sh"
    echo ""

else
    echo "❌ Error: Output file not created"
    exit 1
fi

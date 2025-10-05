CONFIG_PATH="configs/infer/phase_segmentation/ms_tcn.yaml"
TRAINING_DIR="$HOME/.opticai-data/raw/cataract-21/training"
VALIDATION_DIR="$HOME/.opticai-data/raw/cataract-21/validation"

process_videos() {
    local dir="$1"
    local split_name="$2"
    
    echo "Processing $split_name videos in: $dir"
    
    find "$dir" -name "*.mp4" | while read -r video_path; do
        video_name=$(basename "$video_path")
        
        echo "Processing: $video_name"
        
        # Run inference
        python -m scripts.infer_video --config "$CONFIG_PATH" --video "$video_path"
        
        # Check if inference was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed: $video_name"
        else
            echo "✗ Failed to process: $video_name"
        fi
        
        echo "---"
    done
}

echo "Starting batch inference on Cataract-21 dataset..."
echo "Config: $CONFIG_PATH"
echo ""

process_videos "$TRAINING_DIR" "training"

process_videos "$VALIDATION_DIR" "validation"

echo "Batch inference complete!"
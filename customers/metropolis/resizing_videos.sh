for file in /home/arslana/codes/cosmos-predict2/datasets/dbq/raw/*.mp4; do
    filename=$(basename "$file")
    ffmpeg -i "$file" -vf "scale=1280:704:force_original_aspect_ratio=decrease,pad=1280:704:(ow-iw)/2:(oh-ih)/2:black" -c:a copy "/home/arslana/codes/cosmos-predict2/datasets/dbq/train/videos/$filename"
done

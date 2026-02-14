#!/bin/bash
set -e

# Base Directory
PROJECT_DIR="/Users/yuanweize/我的文档/服务器/GITHUB/EY_Challenge_2026"
RESOURCE_DIR="$PROJECT_DIR/resources"

# Create Directories
mkdir -p "$RESOURCE_DIR/docs"
mkdir -p "$RESOURCE_DIR/media"
mkdir -p "$RESOURCE_DIR/data"
mkdir -p "$RESOURCE_DIR/code"
mkdir -p "$RESOURCE_DIR/packages"

cd "$RESOURCE_DIR"

echo "Downloading Documentation..."
curl -L -o "docs/Participant_Guidance.pdf" "https://euwpbww002sta01.blob.core.windows.net/admin-files/2026_EY_AI_%26_Data_Challenge_Participant_Guidance.pdf?sp=r&st=2026-01-16T13:33:15Z&se=2027-01-16T21:48:15Z&spr=https&sv=2024-11-04&sr=b&sig=b1sp0jP%2FFA4ayq1Qpy0ctjOzdECsMO1xDrFGKGrPFDw%3D"

echo "Downloading Media (Videos)..."
curl -L -o "media/Orientation_Session.mp4" "https://challenge.ey.com/api/v1/storage/admin-files/1244834263729444-69846c50df5bdbd5ef85aa21-2026_orientation_session.mp4"
curl -L -o "media/How_to_Get_Started.mp4" "https://euwpbww002sta01.blob.core.windows.net/admin-files/How%20to%20get%20started.mp4?sp=r&st=2026-01-14T13:58:26Z&se=2027-01-14T22:13:26Z&spr=https&sv=2024-11-04&sr=b&sig=%2F1nGmaqM1EqW3R%2BPb2DzoVie9Z1FTfQiNTgo1Ju8MlY%3D"
curl -L -o "media/Tips_for_Success.mp4" "https://euwpbww002sta01.blob.core.windows.net/admin-files/Tips%20for%20success.mp4?sp=r&st=2026-01-20T10:20:32Z&se=2027-01-20T18:35:32Z&spr=https&sv=2024-11-04&sr=b&sig=btLS2ePU0ojhjiqAoIY%2BYHti8QsVgiBit8qyHrXQn%2FU%3D"

echo "Downloading Datasets..."
curl -L -o "data/water_quality_training_dataset.csv" "https://euwpbww002sta01.blob.core.windows.net/admin-files/water_quality_training_dataset.csv?sp=r&st=2026-01-14T13:41:13Z&se=2027-01-14T21:56:13Z&spr=https&sv=2024-11-04&sr=b&sig=WQB2Sxa%2Bgr1FYb80%2BpTcyMm96BHCgYWSkw0O%2BXBYKO0%3D"
curl -L -o "data/submission_template.csv" "https://euwpbww002sta01.blob.core.windows.net/admin-files/submission_template.csv?sp=r&st=2026-01-14T13:50:29Z&se=2027-01-14T22:05:29Z&spr=https&sv=2024-11-04&sr=b&sig=ailUoFpaXTOgstl9kztrUmY8H5Unatf2pwUn3MfUPWA%3D"

echo "Downloading Code & Packages..."
curl -L -o "packages/Snowflake_Notebooks_Package.zip" "https://challenge.ey.com/api/v1/storage/admin-files/14777392681110868-698074f7b3106d93f13bbe08-Snowflake%20Notebooks%20Package.zip"
curl -L -o "packages/Jupyter_Notebook_Package.zip" "https://challenge.ey.com/api/v1/storage/admin-files/6105857096326892-698074de4d5339bd639a4e63-Jupyter%20Notebook%20Package.zip"
curl -L -o "code/snowflake_setup.sql" "https://euwpbww002sta01.blob.core.windows.net/admin-files/snowflake_setup.sql?sp=r&st=2026-01-16T13:34:37Z&se=2027-01-16T21:49:37Z&spr=https&sv=2024-11-04&sr=b&sig=0S80JXeWv5WpkzH3FQzwADfnH8JCSbhqWHehLZVy6AA%3D"
curl -L -o "code/Benchmark_Model_Notebook.ipynb" "https://challenge.ey.com/api/v1/storage/admin-files/5073893047830758-698074304d5339bd639a4714-Benchmark_Model_Notebook.ipynb"
curl -L -o "code/Landsat_Data_Extraction.ipynb" "https://euwpbww002sta01.blob.core.windows.net/admin-files/Landsat_Data_Extraction_Notebook.ipynb?sp=r&st=2026-01-14T13:55:37Z&se=2027-01-14T22:10:37Z&spr=https&sv=2024-11-04&sr=b&sig=ffpCYOAQVyc3xvbVcJqkDoSxeWrQ%2B980HCYNZypPkkM%3D"
curl -L -o "code/TerraClimate_Data_Extraction.ipynb" "https://euwpbww002sta01.blob.core.windows.net/admin-files/TerraClimate_Data_Extraction_Notebook.ipynb?sp=r&st=2026-01-14T13:56:53Z&se=2027-01-14T22:11:53Z&spr=https&sv=2024-11-04&sr=b&sig=1bC%2FqCSuRYwUhVLsOBMRhvptmeEVM3zvzzUs4pH7IGo%3D"

echo "Download Complete."
ls -R "$RESOURCE_DIR"

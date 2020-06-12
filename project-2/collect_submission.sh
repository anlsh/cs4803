rm -f hw2.zip
zip -r hw2.zip . -x "*.ipynb_checkpoints*" "*README.md" "*collect_submission.sh" ".env/*" "*.pyc" "__pycache__/*" "*.git*" "cs7643/datasets*"

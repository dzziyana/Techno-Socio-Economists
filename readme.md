# Setup

Run all notebooks in order (05 must precede 04 so the verifiability parquet exists):

```bash
cd ./rumors && jupyter nbconvert --to notebook --execute --inplace \
  01_data_wrangling.ipynb \
  02_cascade_metrics.ipynb \
  03_statistical_comparison.ipynb \
  05_verifiability.ipynb \
  04_presentation_figures.ipynb
```

from extractor_regr_paper_descriptors import run_regression_from_features_csv

if __name__ == "__main__":
    run_regression_from_features_csv(
        features_csv="results/indigo_aryl_alkyl/indigo_aryl_alkyl_features.csv",
        target="ln(kobs)_MeCN",
        max_features=5,
        output_prefix="results/indigo_aryl_alkyl/indigo_aryl_alkyl_rerun_clean",
        category_name="N-aryl-N′-alkylindigo",
        open_browser=False,
    )
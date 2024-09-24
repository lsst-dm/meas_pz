#!/usr/bin/env bash
butler remove-collections --no-confirm ../ci_hsc_gen3/DATA u/testing/pz_rail_testing
butler remove-runs --no-confirm ../ci_hsc_gen3/DATA "u/testing/pz_rail_testing*"
butler remove-runs --no-confirm ../ci_hsc_gen3/DATA "u/testing/pz_models"
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pz_estimate_knn
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pz_knn_config
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pz_knn_log
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pz_knn_metadata
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pz_estimate_trainz
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pz_trainz_config
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pz_trainz_log
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pz_trainz_metadata
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pzModel_knn
butler remove-dataset-type ../ci_hsc_gen3/DATA/ pzModel_trainz

#!/usr/bin/env bash
butler remove-collections --no-confirm $CI_IMSIM_DIR/DATA u/${USER}/pz_rail_testing
butler remove-runs --no-confirm $CI_IMSIM_DIR/DATA "u/${USER}/pz_rail_testing*"
butler remove-runs --no-confirm $CI_IMSIM_DIR/DATA "u/${USER}/pz_models"
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pz_estimate_knn
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pz_knn_config
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pz_knn_log
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pz_knn_metadata
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pz_estimate_trainz
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pz_trainz_config
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pz_trainz_log
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pz_trainz_metadata
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pzModel_knn
butler remove-dataset-type $CI_IMSIM_DIR/DATA/ pzModel_trainz

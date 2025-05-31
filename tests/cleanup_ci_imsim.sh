#!/usr/bin/env bash
butler remove-collections --no-confirm $CI_IMSIM_DIR/DATA "u/${USER}/pz_rail_testing"
butler remove-runs --no-confirm $CI_IMSIM_DIR/DATA "u/${USER}/pz_rail_testing*"
butler remove-runs --no-confirm $CI_IMSIM_DIR/DATA "u/${USER}/pz_models"

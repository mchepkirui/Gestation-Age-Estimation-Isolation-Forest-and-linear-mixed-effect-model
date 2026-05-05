# =============================================================================
# Gestational Age Estimation Pipeline — Stage 0: Data Consolidation
# =============================================================================
# Repository : mimba-ga-estimation
# Script     : 00_data_consolidation.R
# Authors    : MiMBa Pregnancy Registry Research Team
# Affiliation: KEMRI / Liverpool School of Tropical Medicine
# Data source: MiMBa Pregnancy Registry, Homa Bay County, Kenya
#
# Description
# -----------
# This script implements Stage 0 (Data Consolidation) of the three-stage
# gestational age (GA) estimation pipeline. It ingests validated pregnancy
# episode records linked across multiple facility registers (ANC, maternity,
# EMR), applies programmatic clinical plausibility rules to exclude impossible
# observations, reshapes the data into a long format with one GA measurement
# per row, engineers all features required by the Isolation Forest model
# (Stage 1), and exports a clean CSV ready for anomaly detection.
#
# Clinical motivation
# -------------------
# The MiMBa Registry links records across 42 healthcare facilities in Homa Bay
# County, Kenya. Each pregnancy may have GA measurements from up to nine
# distinct method types: self-reported LMP, ANC-derived GA, fundal height (FH),
# Ballard score (BS), foot length (FL), and first-, second-, or third-trimester
# ultrasound. These measurements are stored in wide format (one column per
# method) and must be reshaped, validated, and feature-engineered before
# anomaly detection.
#
# Workflow (this script)
# ----------------------
# Step 1  Data ingestion         : Load the validated linked episode RDS file.
# Step 2  Outcome filtering      : Exclude discordant, ambiguous, and
#                                  administratively incomplete outcomes.
# Step 3  Wide-to-long reshape   : Pivot from one-row-per-pregnancy to
#                                  one-row-per-GA-measurement.
# Step 4  GA feature engineering : Compute GA at EDD, GA at delivery,
#                                  EDD–delivery difference, z-score, and
#                                  outcome-specific deviation features.
# Step 5  Physiological bounds   : Apply hard GA bounds per outcome category
#                                  (e.g., term birth: 37–44 weeks) and
#                                  reclassify miscarriage/stillbirth by GA.
# Step 6  Source harmonisation   : Collapse granular LMP source labels into
#                                  canonical categories; assign trimester-
#                                  specific ultrasound levels using GA at scan.
# Step 7  Method-level reliability: Assign method-specific standard deviation
#                                  priors (from published literature) and
#                                  reliability ranks for downstream use.
# Step 8  Feature scaling        : Compute within-pregnancy GA z-scores and
#                                  outcome-level scaled gestation (z_gestation).
# Step 9  Summary tables         : Generate method-by-outcome frequency tables
#                                  and export to Excel.
# Step 10 CSV export             : Write the feature-engineered long-format
#                                  dataset for input to Stage 1 (Isolation
#                                  Forest in Python).

# Inputs
# ------
# all_episodes_validated_dataframe.rds
#     Validated linked episode dataframe from the MiMBa Registry linkage
#     pipeline. One row per pregnancy episode; wide format with one column
#     per GA estimation method.
#

# 00-packages.R
#     Package loading script (sourced at startup).
#     Must load: tidyverse, data.table, janitor, writexl, openxlsx, Hmisc

#
# Outputs
# -------
# all_records_for_pythonIF.csv   — Feature matrix for Stage 1

#
# Dependencies
# ------------
# tidyverse, data.table, janitor, writexl, openxlsx, Hmisc
# All loaded via 00-packages.R
#
# Usage
# -----
# Rscript 00_data_consolidation.R
#   or open in RStudio and run interactively.
#   Ensure the working directory contains all input files listed above.
#


# =============================================================================
# 0. Packages and global settings
# =============================================================================

source("00-packages.R")
# Loads all required packages and defines:
#   k_vars           — character vector of key column names to retain
#   lmp_vars         — character vector of wide-format LMP column names
#   lmp_selfreported — character vector of self-reported LMP source labels
#   lmp_ancga        — character vector of ANC-derived GA source labels
#   lmp_anc          — character vector of ANC visit LMP source labels
#   %nin%            — negation of %in%


# =============================================================================
# 1. Data ingestion
# =============================================================================

message("[Stage 0] Loading validated episode data ...")
episodes <- readRDS("all_episodes_validated_dataframe.rds")
message("  Rows loaded : ", nrow(episodes))

# Standardise empty strings to NA for consistent downstream filtering
episodes[episodes == ""] <- NA


# =============================================================================
# 2. Harmonise ultrasound GA field
# =============================================================================
# ga_usall is the primary GA-at-scan field. If it is missing but the
# alternative field us_ga_wks is available, use that value.

epd <- episodes %>%
  mutate(
    ga_usall = case_when(
      is.na(ga_usall) & !is.na(us_ga_wks) ~ us_ga_wks,
      TRUE ~ ga_usall
    )
  )


# =============================================================================
# 3. Column selection and initial filtering
# =============================================================================

# Retain only the columns required for GA estimation; rename LMP step column
# to sd_method (GA dating method indicator from ANC step data).
# Exclude records with missing pregnancy end date or EDD, as these cannot
# contribute a calculable GA estimate.

epi_dt <- epd %>%
  select(mother_id, ga_usall, all_of(k_vars)) %>%
  rename(sd_method = lmp_step) %>%
  filter(!is.na(preg_end), !is.na(edd))

message("  Records after initial filter : ", nrow(epi_dt))


# =============================================================================
# 4. Outcome filtering — exclude discordant and ambiguous categories
# =============================================================================
# The following outcome categories are excluded because they are either
# administratively incomplete, clinically ambiguous (e.g., mixed twin
# outcomes), or represent pregnancies whose GA profile is not comparable
# to the four primary outcome categories used in the model.

EXCLUDED_OUTCOMES <- c(
  "Other",
  "Other, Live birth",
  "Still pregnant",
  "Live birth, Stillbirth - Fresh",
  "Live birth, Stillbirth - Macerated",
  "Live birth, Stillbirth - Unspecified",
  "Induced abortion",
  "Unknown",
  "Pregnancy loss - Unspecified"
)

epi_dt1 <- epi_dt %>%
  filter(moutcome %nin% EXCLUDED_OUTCOMES) %>%
  mutate(
    # Collapse all stillbirth sub-types into a single "Stillbirth" category
    moutcome = case_when(
      str_detect(moutcome, "Stillbirth") ~ "Stillbirth",
      TRUE ~ moutcome
    )
  )

message("Records after outcome filter : ", nrow(epi_dt1))

# =============================================================================
# 5. Date coercion
# =============================================================================
# Convert all date-carrying columns (k_vars minus character columns) to
# R Date objects. This must be done before any difftime() calculations.

date_vars <- setdiff(k_vars, c("lmp_step", "moutcome"))
epi_dt1[, date_vars] <- lapply(date_vars, function(x) as.Date(epi_dt1[[x]]))


# =============================================================================
# 6. Wide-to-long reshape
# =============================================================================
# Each pregnancy has one column per GA estimation method (lmp_vars). Pivoting
# to long format produces one row per (pregnancy × method) combination, which
# is the unit of analysis for the Isolation Forest and LMM.
# Records with missing LMP dates are dropped (no GA can be computed).

dt_eddy <- epi_dt1 %>%
  pivot_longer(
    cols      = all_of(lmp_vars),
    names_to  = "lmp_type",
    values_to = "lmp_date"
  ) %>%
  filter(!is.na(lmp_date)) %>%
  rename(lmp = lmp_date)

message(" Long-format records (method × pregnancy) : ", nrow(dt_eddy))


# =============================================================================
# 7. GA feature engineering
# =============================================================================
# Derive the features used by the Isolation Forest for anomaly detection.
# All GA values are expressed in weeks for clinical interpretability.
#
#   ga_edd        — GA implied by the estimated date of delivery (weeks)
#   ga_del        — GA at the actual delivery/outcome date (weeks)
#   edd_del_diff  — Absolute difference between EDD and delivery date (weeks)
#   s_eddGA       — Standard EDD gestation (40 weeks, appended as a constant)
#   edd_diff      — Deviation of ga_edd from the 40-week standard
#   ga_at_birth   — GA from registered pregnancy start date to delivery (weeks)

dt_eddy <- dt_eddy %>%
  mutate(
    ga_edd       = ceiling(as.numeric(difftime(edd,      lmp,        units = "weeks"))),
    ga_del       = ceiling(as.numeric(difftime(preg_end, lmp,        units = "weeks"))),
    edd_del_diff = abs(as.numeric(difftime(preg_end,     edd,        units = "weeks"))),
    s_eddGA      = 40,
    edd_diff     = abs(s_eddGA - ga_edd),
    ga_at_birth  = abs(as.numeric(difftime(preg_end, preg_start,     units = "weeks")))
  )


# =============================================================================
# 8. Physiological plausibility filter (hard bounds)
# =============================================================================
# Exclude records where GA at delivery falls outside the physiologically
# possible range for any outcome (< 5 weeks or > 44 weeks). This removes
# date entry errors and impossible values before outcome-specific rules.

dt_eddy2 <- dt_eddy %>%
  filter(!(ga_del > 44 | ga_del < 5)) %>%
  rename(lmp_source = lmp_type)

message("  Records after hard GA bounds : ", nrow(dt_eddy2))


# =============================================================================
# 9. Outcome reclassification and outcome-specific bounds
# =============================================================================
# Reclassify miscarriage/stillbirth based on GA at birth (WHO threshold: 28
# weeks). Apply outcome-specific GA bounds and flag out-of-bounds records.

dt_eddy3 <- dt_eddy2 %>%
  mutate(
    # WHO-based reclassification: stillbirths < 28 weeks → miscarriage, and vice versa
    moutcome = case_when(
      moutcome == "Stillbirth" & ga_at_birth <  28 ~ "Miscarriage",
      moutcome == "Miscarriage" & ga_at_birth >= 28 ~ "Stillbirth",
      TRUE ~ moutcome
    ),
    # Create the four-level outcome category used throughout the pipeline
    outcomef = case_when(
      moutcome == "Live birth" & ga_at_birth <  37 ~ "Live_Birth_Preterm",
      moutcome == "Live birth" & ga_at_birth >= 37 ~ "Live_Birth_Term",
      TRUE ~ moutcome
    )
  ) %>%
  mutate(
    # Flag records whose GA at delivery falls outside the expected range
    # for their outcome category — a simple programmatic cleaning rule
    # that precedes the Isolation Forest model.
    isOutBounds = case_when(
      outcomef == "Stillbirth"         & (ga_del <  28 | ga_del >  44) ~ TRUE,
      outcomef == "Live_Birth_Preterm" & (ga_del <  28 | ga_del >= 37) ~ TRUE,
      outcomef == "Live_Birth_Term"    & (ga_del <  37 | ga_del >  44) ~ TRUE,
      outcomef == "Miscarriage"        & (ga_del >= 28 | ga_del <   5) ~ TRUE,
      outcomef == "ectopic"            & (ga_del <   6 | ga_del >  12) ~ TRUE,
      TRUE ~ FALSE
    )
  )

dx_final <- dt_eddy3 %>% filter(!isOutBounds)
message("  Records after outcome-specific bounds : ", nrow(dx_final))


# =============================================================================
# 10. Factor coercion
# =============================================================================

ch_vars <- c("moutcome", "lmp_source", "sd_method", "outcomef")
dx_final[, ch_vars] <- lapply(ch_vars, function(x) as.factor(dx_final[[x]]))

setDT(dx_final)
summary(dx_final)


# =============================================================================
# 11. Summary Table A — method frequency by outcome (pre-exclusion)
# =============================================================================

summary_table_A <- dx_final %>%
  group_by(sd_method, outcomef) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(outcomef) %>%
  mutate(
    N       = sum(n),
    percent = round((n / N) * 100, 1),
    summary = paste0(n, "/", N, " (", percent, "%)")
  ) %>%
  select(sd_method, outcomef, summary) %>%
  pivot_wider(names_from = outcomef, values_from = summary) %>%
  clean_names()

print(summary_table_A)
write_xlsx(summary_table_A, "lmp_summary_by_outcome.xlsx")


# =============================================================================
# 12. Method exclusions and deduplication
# =============================================================================
# Exclude:
#   (a) Outcome-derived GA sources (iacrf, postnatal outcome records, maternity
#       outcome records) 
#   (b) Ballard score and foot length in stillbirths — these neonatal assessment
#       methods are not applicable to intrauterine fetal deaths.
#   (c) Foot length in miscarriages — not applicable at early gestations.
#   (d) Duplicate ANC-visit LMP values within the same pregnancy — retain only
#       the first occurrence per pregnancy × source combination.
#   (e) Require >= 3 method records per pregnancy for model identifiability.

OUTCOME_DERIVED_SOURCES <- c(
  "lmp_iacrf", "lmp_POutcome", "lmp_MATOutcome", "lmp_GAMaternity"
)

ds <- dx_final %>%
  filter(lmp_source %nin% OUTCOME_DERIVED_SOURCES) %>%
  filter(
    !(lmp_source == "lmp_ballard_final"    & outcomef == "Stillbirth"),
    !(lmp_source == "lmp_footlength_final" & outcomef == "Stillbirth"),
    !(lmp_source == "lmp_footlength_final" & outcomef == "Miscarriage")
  ) %>%
  group_by(mother_id) %>%
  filter(!(lmp_source %in% lmp_anc & duplicated(lmp))) %>%
  ungroup()

ds1 <- ds %>%
  group_by(mother_id) %>%
  filter(n() >= 3) %>%
  ungroup()

message("  Records after method exclusions (n >= 3) : ", nrow(ds1))


# =============================================================================
# 13. Within-pregnancy GA z-score
# =============================================================================
# For each pregnancy, compute the mean and SD of ga_del across all retained
# method records. The within-pregnancy z-score quantifies how far each
# method's GA estimate deviates from the pregnancy-level average — a key
# feature for identifying anomalous measurements in the Isolation Forest.
# Records with no within-pregnancy variation (sd = 0, implying only one
# distinct GA value) are excluded as they contribute no discriminative signal.

ds2 <- ds1 %>%
  group_by(mother_id) %>%
  mutate(
    ga_mean_preg      = mean(ga_del, na.rm = TRUE),
    ga_sd_preg        = sd(ga_del, na.rm = TRUE),
    ga_diff_from_mean = abs(ga_del - ga_mean_preg),
    ga_zscore         = ifelse(
      ga_sd_preg > 0,
      (ga_del - ga_mean_preg) / ga_sd_preg,
      NA_real_
    )
  ) %>%
  filter(!is.na(ga_zscore)) %>%
  ungroup()

message("  Records with computable within-pregnancy z-score : ", nrow(ds2))


# =============================================================================
# 14. Outcome-specific expected GA range features
# =============================================================================
# Append clinically expected GA bounds per outcome. These define the
# physiologically plausible range and are used to compute ga_deviation —
# the distance of each measurement from the nearest bound — which is a
# direct feature for the Isolation Forest.

ds3 <- ds2 %>%
  mutate(
    expected_ga_min = case_when(
      outcomef == "Ectopic"            ~  6,
      outcomef == "Miscarriage"        ~  5,
      outcomef == "Stillbirth"         ~ 28,
      outcomef == "Live_Birth_Preterm" ~ 28,
      outcomef == "Live_Birth_Term"    ~ 37,
      TRUE ~ NA_real_
    ),
    expected_ga_max = case_when(
      outcomef == "Ectopic"            ~ 12.0,
      outcomef == "Miscarriage"        ~ 27.9,
      outcomef == "Stillbirth"         ~ 44.0,
      outcomef == "Live_Birth_Preterm" ~ 36.9,
      outcomef == "Live_Birth_Term"    ~ 44.0,
      TRUE ~ NA_real_
    )
  ) %>%
  mutate(
    # Distance to boundary: 0 if within expected range, positive if outside
    ga_gap_to_max = expected_ga_max - ga_del,
    ga_gap_to_min = ga_del - expected_ga_min
  )

# Encode outcome as numeric for Isolation Forest (Python sklearn requirement)
ds4 <- ds3 %>%
  mutate(
    outcomef = as.factor(outcomef),
    outcome  = as.numeric(outcomef)
  )


# =============================================================================
# 15. LMP source harmonisation and ultrasound trimester classification
# =============================================================================
# Collapse granular source labels into canonical categories matching the LMM
# factor levels. Assign trimester-specific labels to ultrasound records based
# on GA at scan (ga_usall). 

ds_if <- ds4 %>%
  mutate(
    lmp_source = case_when(
      lmp_source %in% lmp_selfreported ~ "self_reported",
      lmp_source %in% lmp_ancga        ~ "anc_ga_date",
      TRUE ~ lmp_source
    ),
    ga_usall = round(as.numeric(ga_usall), 4)
  ) %>%
  mutate(
    us_trimester = case_when(
      !is.na(ga_usall) & ga_usall >  0 & ga_usall < 14 ~ "1st Trimester",
      !is.na(ga_usall) & ga_usall >= 14 & ga_usall < 28 ~ "2nd Trimester",
      !is.na(ga_usall) & ga_usall >= 28                  ~ "3rd Trimester",
      TRUE ~ NA_character_
    )
  ) %>%
  mutate(
    lmp = as.Date(lmp),
    # Trimester-specific ultrasound label for use in LMM factor levels
    source_lev = case_when(
      lmp_source == "lmp_ultrasound" & us_trimester == "1st Trimester" ~ "lmp_ultrasound_1",
      lmp_source == "lmp_ultrasound" & us_trimester == "2nd Trimester" ~ "lmp_ultrasound_2",
      lmp_source == "lmp_ultrasound" & us_trimester == "3rd Trimester" ~ "lmp_ultrasound_3",
      TRUE ~ lmp_source
    )
  ) %>%
  select(-lmp_source) %>%
  rename(lmp_source = source_lev)


# =============================================================================
# 17. GA deviation feature
# =============================================================================
# ga_deviation quantifies how far a measurement falls outside the outcome-
# specific expected GA range. It is 0 for measurements within the expected
# range and positive for those outside. This is a primary Isolation Forest
# feature for identifying physiologically implausible records.

ds_if2 <- ds_if1 %>%
  mutate(
    ga_deviation = case_when(
      ga_del < expected_ga_min ~ expected_ga_min - ga_del,
      ga_del > expected_ga_max ~ ga_del - expected_ga_max,
      TRUE ~ 0
    )
  ) %>%
  filter(!is.na(expected_ga_min))


# =============================================================================
# 18. Outcome-level GA scaling (z_gestation)
# =============================================================================
# Scale GA at delivery within each outcome stratum. This normalised metric
# captures how anomalous a GA value is relative to the distribution of
# GA values for the same outcome category — a complementary signal to the
# within-pregnancy z-score computed in Step 13.
#
# Additional binary implausibility flags are added for records that are:
#   - extreme z-gestation outliers (|z| > 3)
#   - have EDD or delivery date before the conception date (impossible)

ds_if3 <- ds_if2 %>%
  as.data.frame() %>%
  group_by(outcomef) %>%
  mutate(z_gestation = as.numeric(scale(ga_del))) %>%
  ungroup() %>%
  mutate(
    edd_before_lmp      = as.numeric(edd      < lmp),
    delivery_before_lmp = as.numeric(preg_end < lmp),
    implausible_ga      = as.numeric(abs(z_gestation) > 3)
  ) %>%
  filter(!is.na(edd)) %>%
  filter(outcomef != "Ectopic") %>%     # too few records for stable modelling
  ungroup() %>%
  select(
    mother_id, preg_start, preg_end, lmp_source, source_sd,
    lmp, edd, ga_edd, ga_del, outcomef, expected_ga_min, outcome,
    expected_ga_max, ga_zscore, ga_deviation, z_gestation,
    edd_before_lmp, delivery_before_lmp, implausible_ga
  )

message("\n  Final feature matrix dimensions : ", nrow(ds_if3),
        " rows × ", ncol(ds_if3), " columns")


# =============================================================================
# 19. Export feature matrix for Stage 1
# =============================================================================

write.csv(ds_if3, "all_records_for_pythonIF_March2026.csv", row.names = FALSE)
message("  Saved: all_records_for_pythonIF_March2026.csv")


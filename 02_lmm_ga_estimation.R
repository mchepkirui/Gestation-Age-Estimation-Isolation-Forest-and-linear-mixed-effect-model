# =============================================================================
# Gestational Age Estimation Pipeline — Stage 2: Statistical Integration
# =============================================================================

# Description
# -----------
# This script implements Stage 2 (Statistical Integration) of the two-stage
# gestational age (GA) estimation pipeline. It ingests the anomaly-cleaned
# dataset produced by Stage 1 (01_anomaly_detection_isolation_forest.py) and
# fits a linear mixed-effects model (LMM) to derive a single, optimised
# 'best GA' estimate for each pregnancy.
#

# -------------------
#The LMM:
#
#   (a) Treats GA measurement as the dependent variable.
#   (b) Includes GA_method and birth_outcome as fixed effects to account
#       for method-specific biases and outcome-related physiology.
#   (c) Includes a random intercept per pregnancy_id, allowing each pregnancy
#       to have its own baseline GA while sharing population-level parameter
#       estimates — a form of partial pooling. 
#   (d) Generates predictions anchored to first-trimester ultrasound (US1),
#       the most clinically reliable GA reference in this dataset.
#
# Workflow (this script)
# ----------------------
# Step 1  Data ingestion         : Load clean_data_for_lmm.csv (Stage 1
#                                  output) and apply within-pregnancy deduplication
#                                  of self-reported LMP records.
# Step 2  Feature engineering    : Compute GA at delivery (days) as the
#                                  difference between delivery date and each
#                                  method's conception date estimate.
# Step 3  Reference setting      : Set US1 (first-trimester ultrasound) as
#                                  the reference level for GA_method, and
#                                  Live_Birth_Term as the reference outcome.
# Step 4  LMM fitting            : Fit lmer(GA_birth ~ GA_method +
#                                  birth_outcome + (1|pregnancy_id)).
# Step 5  Best-GA prediction     : Predict GA for each pregnancy under the
#                                  US1 reference, incorporating pregnancy-
#                                  specific random effects.
# Step 6  Conception date        : Back-calculate conception date from the
#                                  predicted GA and delivery date.
# Step 7  Bias assessment        : Compute mean bias and 95% limits of
#                                  agreement (Bland-Altman) for each method
#                                  against the model consensus.
# Step 8  Visualisation          : Produce a three-panel shrinkage plot for
#                                  three example pregnancies, illustrating
#                                  how the model pulls method-specific dates
#                                  toward the consensus estimate.
# Step 9  Output export          : Save predicted GA data as .rds and figures
#                                  as high-resolution JPEG and PDF.
#
# Inputs
# ------
# clean_data_for_lmm.csv  (produced by Stage 1)
#   Required columns:
#     mother_id    — Unique pregnancy identifier
#     lmp          — Conception date estimated by each method (Date)
#     lmp_source   — GA estimation method
#                    (LMP | M_LMP | ANC_GA | FH | BS | FL | US1 | US2 | US3)
#     outcomef     — Pregnancy outcome
#                    (Live_Birth_Term | Live_Birth_Preterm |
#                     Miscarriage | Stillbirth)
#     preg_end     — Date of pregnancy outcome (Date)
#
# Outputs
# -------
# new_cdate_march2026.rds          — Per-pregnancy best-GA predictions
#
# Dependencies
# ------------
# merTools, patchwork, tidyverse, jtools, lmerTest, rlang,
# ggplot2, ggrepel, cowplot, showtext
#
# Install: install.packages(c("merTools","patchwork","tidyverse","jtools",
#          "lmerTest","rlang","ggplot2","ggrepel","cowplot","showtext"))
#
# Usage
# -----
# Rscript 02_lmm_ga_estimation.R
#   or open in RStudio and run interactively.
#   Update INPUT_FILE below to point to Stage 1 output.
#


# =============================================================================
# 0. Libraries
# =============================================================================

library(merTools)   # predictInterval() for LMM prediction intervals
library(patchwork)  # composing multi-panel ggplot figures
library(tidyverse)  # dplyr, tidyr, ggplot2, lubridate, readr
library(jtools)     # summ() — formatted model summaries
library(lmerTest)   # lmer() with Satterthwaite df and p-values
library(rlang)      # tidy evaluation helpers
library(ggplot2)    # primary plotting grammar
library(ggrepel)    # non-overlapping text labels
library(cowplot)    # theme_cowplot(), plot_grid()
library(showtext)   # custom fonts in ggplot2

# Embed Times New Roman in all ggplot2 output.
# times.ttf must be present in the working directory or system font path.
showtext_auto()
font_add("Times New Roman", "times.ttf")

options(scipen = 999)  # suppress scientific notation in console output


# =============================================================================
# 1. Configuration
# =============================================================================

INPUT_FILE  <- "Anomaly results/clean_data_for_modelling.csv"
OUTPUT_RDS  <- "new_cdate_march2026.rds"

# GA estimation method labels in the desired display order.
# Must match the factor levels after coercion from lmp_source.
GA_METHOD_LEVELS <- c("ANC_GA", "FH", "BS", "FL", "US1", "US2", "US3",
                       "M_LMP", "LMP")

# Reference levels for fixed-effect contrasts
REF_METHOD  <- "US1"              # first-trimester ultrasound
REF_OUTCOME <- "Live_Birth_Term"  # most clinically expected outcome


# =============================================================================
# 2. Helper functions
# =============================================================================

#' Mode of a vector (returns most frequent value; ties broken by first
#' occurrence)
mode_func <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}


# =============================================================================
# 3. Data ingestion and cleaning
# =============================================================================

# --- 3.1 Load Stage 1 output ---
message("\n[Stage 2] Loading clean dataset from: ", INPUT_FILE)
lmm_dt <- read.csv(INPUT_FILE)
message("  Rows loaded : ", nrow(lmm_dt))

# --- 3.2 Deduplicate self-reported LMP records ---
# A single pregnancy may have multiple self-reported LMP records if the mother
# attended ANC more than once and reported slightly different dates. We retain
# only the modal (most common) self-reported LMP per pregnancy; all other
# source types are retained in full.
#
# We then require each pregnancy to have at least 3 distinct method records
# so the random intercept can be identified.

lmm_dt_clean <- lmm_dt %>%
  group_by(mother_id) %>%
  mutate(
    self_reported_mode = ifelse(
      any(lmp_source == "self_reported"),
      mode_func(lmp[lmp_source == "self_reported"]),
      NA_character_
    )
  ) %>%
  filter(
    lmp_source != "self_reported" | lmp == self_reported_mode
  ) %>%
  # When ties produce duplicate self-reported modal records, keep the first
  group_by(mother_id, lmp_source) %>%
  slice(1) %>%
  ungroup() %>%
  dplyr::select(-self_reported_mode) %>%
  group_by(mother_id) %>%
  filter(n() >= 3) %>%          # require >= 3 method records per pregnancy
  ungroup()

message("  Records after deduplication : ", nrow(lmm_dt_clean))
message("  Unique pregnancies          : ",
        n_distinct(lmm_dt_clean$mother_id))


# =============================================================================
# 4. Feature engineering
# =============================================================================

# --- 4.1 Standardise column names ---
df_x <- lmm_dt_clean %>%
  dplyr::select(mother_id, lmp, outcomef, lmp_source, preg_end) %>%
  rename(
    pregnancy_id   = mother_id,
    GA_method      = lmp_source,   # GA estimation method
    c_date         = lmp,          # conception date estimated by this method
    birth_outcome  = outcomef,     # pregnancy outcome category
    delivery_date  = preg_end      # date of outcome
  )

# --- 4.2 Type coercion ---
mdta <- df_x %>%
  mutate(
    pregnancy_id  = as.factor(pregnancy_id),
    GA_method     = as.factor(GA_method),
    birth_outcome = as.factor(birth_outcome),
    delivery_date = as.Date(delivery_date),
    c_date        = as.Date(c_date)
  )

# --- 4.3 Compute GA at delivery (days) ---
# GA_birth = delivery_date - conception_date_per_method
# This is the response variable in the LMM.
dt1 <- mdta %>%
  mutate(
    GA_birth = as.numeric(difftime(delivery_date, c_date, units = "days"))
  )

# --- 4.4 Set factor levels and reference categories ---
levels(dt1$GA_method) <- GA_METHOD_LEVELS

# US1 (first-trimester ultrasound) is the clinical gold standard reference.
# All method coefficients will be interpreted relative to US1.
dt1$GA_method     <- relevel(dt1$GA_method,     ref = REF_METHOD)
dt1$birth_outcome <- relevel(dt1$birth_outcome, ref = REF_OUTCOME)

message("\n  GA_method levels : ", paste(levels(dt1$GA_method), collapse = ", "))
message("  Outcome levels   : ", paste(levels(dt1$birth_outcome), collapse = ", "))


# =============================================================================
# 5. Linear mixed-effects model
# =============================================================================
#
# Model specification:
#   GA_birth ~ GA_method + birth_outcome + (1 | pregnancy_id)
#
# Fixed effects:
#   GA_method      — systematic bias of each dating method relative to US1
#   birth_outcome  — GA shift attributable to outcome category (e.g., preterm
#                    births have inherently shorter gestations)
#
# Random effect:
#   (1 | pregnancy_id) — random intercept capturing pregnancy-specific
#                        deviations in true gestational age; enables partial
#                        pooling (shrinkage) of method-specific estimates
#                        toward the pregnancy-level consensus

message("\n[Stage 2] Fitting linear mixed-effects model ...")

eDate <- lmer(
  GA_birth ~ GA_method + birth_outcome + (1 | pregnancy_id),
  data = dt1
)

message("\n--- Model summary ---")
print(summary(eDate))


# =============================================================================
# 6. Best-GA prediction (anchored to US1)
# =============================================================================
#
# We predict GA for every pregnancy using GA_method = US1 as the covariate
# value, while retaining the pregnancy-specific random intercept.
# This is equivalent to asking: "what would this pregnancy's GA be if it had
# been measured by first-trimester ultrasound?"

message("\n[Stage 2] Generating best-GA predictions (reference: US1) ...")

preg_df <- dt1 %>%
  group_by(pregnancy_id) %>%
  summarise(
    delivery_date = first(delivery_date),
    birth_outcome = first(birth_outcome),
    .groups = "drop"
  ) %>%
  mutate(
    GA_method = factor(REF_METHOD, levels = levels(dt1$GA_method))
  )

preg_df$GA_best_event <- predict(
  eDate,
  newdata  = preg_df,
  re.form  = ~ (1 | pregnancy_id)  # include pregnancy random effects
)

# Back-calculate the consensus conception date
preg_df$conception_date <- preg_df$delivery_date - preg_df$GA_best_event

message("  Predictions generated for ", nrow(preg_df), " pregnancies.")
message("  Saving predictions to: ", OUTPUT_RDS)
saveRDS(preg_df, OUTPUT_RDS)


# =============================================================================
# 7. Bland–Altman bias assessment
# =============================================================================
#
# For each dating method, compute the mean bias and 95% limits of agreement
# against the model consensus (GA_best_event). A well-calibrated method
# will show a mean bias close to 0; wide limits of agreement indicate high
# within-method variability.

message("\n[Stage 2] Computing Bland-Altman bias by method ...")

dt_with_best <- dt1 %>%
  left_join(
    preg_df %>%
      dplyr::select(pregnancy_id, conception_date, GA_best_event),
    by = "pregnancy_id"
  ) %>%
  mutate(
    diff_method_vs_best = GA_birth - GA_best_event,   # +ve = method overestimates GA
    new_ga = GA_best_event / 7                        # convert days to weeks
  )

bias_by_method <- dt_with_best %>%
  group_by(GA_method) %>%
  summarise(
    mean_bias = round(mean(diff_method_vs_best, na.rm = TRUE), 3),
    sd_bias   = sd(diff_method_vs_best, na.rm = TRUE),
    upper_LOA = mean_bias + 1.96 * sd_bias,
    lower_LOA = mean_bias - 1.96 * sd_bias,
    n         = n(),
    abs_bias  = abs(mean_bias),
    .groups   = "drop"
  )

message("\n--- Bias by method (sorted by absolute bias) ---")
print(bias_by_method %>% arrange(abs_bias))


# =============================================================================
# 8. Shrinkage visualisation
# =============================================================================
#
# To illustrate model behaviour, we plot three example pregnancies side by
# side. For each pregnancy:
#   - Coloured points = method-specific conception date estimates.
#   - Red diamond = model consensus (best) conception date.
#   - Grey segments = shrinkage pull from each method toward the consensus.
#   - Day labels = distance in days from method estimate to consensus.
#   - Red text = consensus date on the x-axis.

message("\n[Stage 2] Preparing shrinkage visualisation ...")

df_rand <- dt_with_best %>%
  filter(pregnancy_id %in% EXAMPLE_PREGNANCIES) %>%
  mutate(
    pregnancy_id = as.character(pregnancy_id),
    pregnancy_id = as.factor(pregnancy_id),
    c_date       = as.Date(c_date,       origin = "1970-01-01"),
    best_c_date  = as.Date(conception_date, origin = "1970-01-01")
  )

# Recode to anonymous labels for the manuscript
levels(df_rand$pregnancy_id) <- c("p1", "p2", "p3")

# --- Prepare plot data ---
plot_data <- df_rand %>%
  filter(pregnancy_id %in% c("p1", "p2", "p3")) %>%
  mutate(
    c_date       = as.Date(c_date),
    best_c_date  = as.Date(best_c_date),
    pregnancy_id = factor(pregnancy_id, levels = c("p1", "p2", "p3"))
  )

# --- Build the plot ---
final_plot <- ggplot(plot_data, aes(y = GA_method)) +

  # Consensus reference line (vertical, per panel)
  geom_vline(
    aes(xintercept = best_c_date),
    color = "#e74c3c", linewidth = 1, alpha = 0.4
  ) +

  # Shrinkage segments: method estimate → consensus
  geom_segment(
    aes(x = c_date, xend = best_c_date, yend = GA_method),
    color = "grey70"
  ) +

  # Method-specific conception date points
  geom_point(
    aes(x = c_date, color = "Method-Specific Date"),
    size = 3
  ) +

  # Consensus (best) conception date points
  geom_point(
    aes(x = best_c_date, color = "Model Consensus (Best)"),
    shape = 18, size = 5
  ) +

  # Day-difference labels midway along each segment
  geom_text(
    aes(
      x     = c_date + (best_c_date - c_date) / 2,
      label = paste0(abs(round(diff_method_vs_best, 1)), "d")
    ),
    vjust = -0.7, size = 3, fontface = "italic"
  ) +

  # Consensus date label on the x-axis
  geom_text(
    aes(
      x     = best_c_date,
      y     = 0.4,
      label = format(best_c_date, "%b %d, %Y")
    ),
    color = "#e74c3c", fontface = "bold", size = 3, vjust = 1
  ) +

  # Facets: one panel per example pregnancy
  facet_wrap(~ pregnancy_id, scales = "free_x", ncol = 3) +

  # Aesthetics
  scale_color_manual(
    name   = "Legend : ",
    values = c(
      "Method-Specific Date"    = "#3498db",
      "Model Consensus (Best)"  = "#e74c3c"
    )
  ) +
  scale_x_date(date_labels = "%b %d") +
  scale_y_discrete(expand = expansion(add = c(1, 0.6))) +

  labs(
    title    = "Linear Mixed Effects Model: Shrinkage of Dating Methods toward Consensus",
    subtitle = paste0(
      "The vertical line represents the model best estimate.\n",
      "Day labels represent the distance from the method-specific ",
      "date to the model estimate."
    ),
    x       = NULL,
    y       = "Method",
    caption = paste0(
      "US1, US2, US3: ultrasound scans in the first, second and third trimester. ",
      "LMP: last menstrual period. ANC GA: gestational age at ANC visit.\n",
      "FH: fundal height. BS: Ballard score. FL: foot length."
    )
  ) +

  theme_bw(base_family = "Times New Roman") +
  theme(
    legend.position      = "top",
    legend.justification = "center",
    plot.title    = element_text(
      hjust = 0.5, face = "bold", size = 16, family = "Times New Roman"
    ),
    plot.subtitle = element_text(
      hjust = 0.5, size = 11, family = "Times New Roman"
    ),
    plot.caption  = element_text(
      hjust = 0, face = "italic", size = 9, family = "Times New Roman"
    ),
    strip.text    = element_text(
      face = "bold", size = 12, family = "Times New Roman"
    )
  )

# --- Preview ---
print(final_plot)


# =============================================================================
# 9. Save figures
# =============================================================================

message("\n[Stage 2] Saving figures ...")

ggsave(
  filename = OUTPUT_JPEG,
  plot     = final_plot,
  device   = "jpeg",
  width    = 12,     # wide for 3 side-by-side panels
  height   = 5.5,
  units    = "in",
  dpi      = 300,    # standard manuscript resolution
  quality  = 95
)
message("  Saved: ", OUTPUT_JPEG)

ggsave(
  filename = OUTPUT_PDF,
  plot     = final_plot,
  width    = 16,
  height   = 8,
  device   = "pdf"
)
message("  Saved: ", OUTPUT_PDF)


# =============================================================================
# 10. Done
# =============================================================================

message("\n[Stage 2] COMPLETE.")
message("  Best-GA predictions : ", OUTPUT_RDS)
message("  Shrinkage plot JPEG : ", OUTPUT_JPEG)
message("  Shrinkage plot PDF  : ", OUTPUT_PDF)
message(
  "\n  Next steps: use preg_df$GA_best_event and preg_df$conception_date ",
  "for downstream\n  classifications (preterm birth, SGA) and ",
  "pharmacovigilance analyses."
)

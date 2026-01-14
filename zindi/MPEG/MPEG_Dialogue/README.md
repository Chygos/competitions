![title_img](mpeg_dialogue_img.png)

# **MPEG-G: Decoding the Dialogue**

This repository contains scripts to the [MPEG-Decoding the Dialogue challenge](https://zindi.africa/competitions/mpeg-g-decoding-the-dialogue) hosted on [Zindi](https://zindi.africa). In this repository, I will be sharing my winning solution to the track 2 challenge.

## Challenge Objective

This challengeaimed to determine host-microbiome interactions in metagenomic data. The challenge was divided into five tracks.

### Summary of Tracks

#### Track 1: Predictive model to predict cytokine levels using microbiome data

- Use predictive models to predict cytokine levels using metagenomic data obtained across body sites.

#### Track 2: Microbe–Cytokine Association Discovery

- Identify key associations between microbial taxa and cytokines. 

Solution is to build a model that identifies and ranks (or confidence scores) important interactions between microbes and cytokines

#### Track 3: Graph-Based Host-Microbiome Network Discovery

- Create a network of microbe-cytokine interactions linked to body sites.

Solution is to model microbe-cytokine interaction entities in a graph. Network should link and interpret microbial taxa to cytokine profiles across participants, captuing body-site-aware and time-dependent interactions

#### Track 4: Latent Health State Discovery via Embeddings

- Derive meaningful low-dimensional representations of host-microbe interactions using unsupervised or contrastive learning. Embeddings should reflect personalised dynamics or immune phenotypes

#### Track 5: Open-Ended Discovery

- Share any interesting insights about microbe-cytokine interactions not mentioned in other tracks


## Approach (Track 2)

I participated in three tracks: 1, 2 and 5. Track 2 solution was dependent on some preprocessing steps from tracks 1 and 5.

### Data Preparation

Dataset was obtained from 16S rRNA data stored in Fastq files

#### Data Cleaning
- Removed invalid bases (bases with Ns, if present)
- Removed bases with low quality scores (below 20) and reads with sequence lengths less than 50
- Created 8-kmer sequences, with no successive skips.
- Aggregated them into counts.
- Selecting samples with matching cytokine levels.

#### Feature Selection

Tracks 2 and 5 depend on feature selection methods applied in track 1.

- In track 1, variance thresholding was applied to filter out low-variance kmer sequences. This reduced the 65,536 unique kmers by about 80%. 

- Remaining kmers in step 1 were further trimmed by applying an elastic linear regression model. Non-zero coefficients (1,492) kmers left were then used for track 5

- In track 5, the Spearman's correlation analysis was applied to select kmers that associate with cytokines. Statistically significant kmers with $\alpha$ < 0.05, and effect size ($\rho$ >= 0.25), were selected.

Correlation method and the Spearman’s correlation method were selected due to the non-linear relationship ability of the Spearman’s correlation and under the assumption that a kmer that associates with a cytokine is capable of eliciting an immune response. Only 262 significant kmers were left.

#### Normalisation

Kmer counts were normalised using the centred-log ratio (CLR) method

$$ CLR = log(1 + counts) − mean(log(1 + counts)) $$


### Microbe-Cytokine Association

To model microbe-cytokine association that outputs a rank and confidence score, we used the bootstrap method to generate resamples of the original data. The confidence scores provide an estimate of the range within which the magnitude of a kmer’s effect on a cytokine is likely to fall under repeated sampling. 

To achieve this, we simulated kmer–cytokine data alongside metabolic and demographic variables—including gender, age, fasting plasma glucose, and body mass index. Metabolic and demographic variables were chosen because in track 5, it was identified that these variables affect cytokine levels.

To increase the consistency of result and produce smooth confidence scores, resampling was done 1000 times and used to compute 95% confidence interval scores. With the boostrap method, it is easier to compute the lower and upper bounds for each coefficient (association).

### Stability scores

To quantify the reliability of the association between a kmer and a cytokine, various stability metrics were computed at each bootstrap to identify kmers or microbes whose effects are not due to sampling error or spurious relationship. These stability scores include: sign stability, coefficient of variation stability, top-N stability and scaled absolute mean coefficient. 

**Sign Stability**

The sign stability reflects directional consistency. It helps to quantify the proportion of times microbe-cytokine association remain the same in sign (direction) at each bootstrap. Low sign stability indicate that the association is less interpretable or potentially spurious.

**Coefficient of Variation (CV) Stability**

The coefficient of variation (CV) stability measures the consistency of the effect size. It is a ratio of the variability (standard deviation) and averag association in each bootstrap. Kmers with low CV stability exhibit low variability relative to their average effect, indicating a more reliable association. This score was converted to have a 0-1 range by taking the adding 1 to the CV score and taking the reciprocal.

**Top-N Stability**

This captures the frequency with which a kmer ranks among the top N (top 100) most influential features, based on their absolute coefficient scores across bootstrap. This reflects relative importance and helps to prioritise kmers that consistently emerge as key contributors to cytokine variation.

**Scaled Absolute Mean Coefficient**

This represents the overall strength of association, independent of direction. Scaling was done to ensure comparability among features and for easy integration with other stability metrics.

**Composite Score**

To construct a composite score, we aggregated the CV and sign stability scores by using a weighted mean. This composite score allows us to rank microbe-cytokine association based on their statistical robustness and biological relevance. It also enables us to filter out unstable associations and focus on reproducible microbe-cytokine relationships that are less likely to be driven by sampling noise. Similarly, the CV and sign stability were used in this computation because it was identified the topN and scaled absolute mean coefficient scores were biased because kmers sequences contain ovelapping gene sequences and the threshold chosen (top 100) was arbitrary.

## Bootstrapping and Modelling

For each boostrap, a simple linear model: a multivariate ridge linear regression model was used. This model was used because the penalty it applies reduces the effect of multicollinearity and also because we assumed a linear relationship to be between a cytokine and microbe, where the increase in microbial count activates the immune system, increasing the production of cytokines. Also, a linear model's coefficients are interpretable and can enable us to quantify the magnitude and direction of a microbe's effect on a cytokine. 

However, since cytokines are affected by some host factors-body site, gender, age, metabolic states-these covariates were included in the model so that the effect obtained represents the marginal contribution of a kmer (microbe) on a cytokine. Additionally, for easy comparison of coefficients, both kmers and cytokine values were standardised so that the interpretation of the effect reflects the expected change of cytokines in standard deviation units with a unit increase in kmer/microbial count.


## Results

Results can be found [here](track2/Track2_report.md) 
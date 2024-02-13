# History

## v0.9.0 - 2024-02-13

This release makes CTGAN sampling more efficient by saving the frequency of each categorical value.

### New Features

* Improve DataSampler efficiency - Issue [#327] ((https://github.com/sdv-dev/CTGAN/issue/327)) by @fealho

## v0.8.0 - 2023-11-13

This release adds a progress bar that will show when setting the `verbose` parameter to `True`
when initializing `TVAE`.

### New Features

* Add verbosity TVAE (progress bar + save the loss values) - Issue [#300]((https://github.com/sdv-dev/CTGAN/issues/300) by @frances-h

## v0.7.5 - 2023-10-05

This release adds a progress bar that will show when setting the `verbose` parameter to True when initializing `CTGAN`. It also removes a warning that was showing.

### Maintenance

* Remove model_missing_values from ClusterBasedNormalizer call - PR [#310](https://github.com/sdv-dev/CTGAN/pull/310) by @fealho
* Switch default branch from master to main - Issue [#311](https://github.com/sdv-dev/CTGAN/issues/311) by @amontanez24
* Remove or implement CTGAN tests - Issue [#312](https://github.com/sdv-dev/CTGAN/issues/312) by @fealho

### New Features

* Add progress bar for CTGAN fitting (+ save the loss values) - Issue [#298](https://github.com/sdv-dev/CTGAN/issues/298) by @frances-h

## v0.7.4 - 2023-07-25

This release adds support for Python 3.11 and drops support for Python 3.7.

### Maintenance

* Why is there an upper bound in the packaging requirement? (packaging<22) - Issue [#276](https://github.com/sdv-dev/CTGAN/issues/276) by @fealho
* Add support for Python 3.11 - Issue [#296](https://github.com/sdv-dev/CTGAN/issues/296) by @fealho
* Drop support for Python 3.7 - Issue [#302](https://github.com/sdv-dev/CTGAN/issues/302) by @fealho

## v0.7.3 - 2023-05-25

This release adds support for Torch 2.0!

### Bugs Fixed

* Torch 2.0 fails with cuda=False - Issue [#288](https://github.com/sdv-dev/CTGAN/issues/288) by @amontanez24

### Maintenance

* Upgrade to torch 2.0 - Issue [#280](https://github.com/sdv-dev/CTGAN/issues/280) by @frances-h

## v0.7.2 - 2023-05-09

This release adds support for Pandas 2.0! It also fixes a bug in the `load_demo` function.

### Bugs Fixed

* load_demo raises urllib.error.HTTPError: HTTP Error 403: Forbidden - Issue [#284](https://github.com/sdv-dev/CTGAN/issues/284) by @amontanez24

### Maintenance

* Remove upper bound for pandas - Issue [#282](https://github.com/sdv-dev/CTGAN/issues/282) by @frances-h

## v0.7.1 - 2023-02-23

This release fixes a bug that prevented the `CTGAN` model from being saved after sampling.

### Bugs Fixed

* Cannot save CTGANSynthesizer after sampling (TypeError) - Issue [#270](https://github.com/sdv-dev/CTGAN/issues/270) by @pvk-developer

## v0.7.0 - 2023-01-20

This release adds support for python 3.10 and drops support for python 3.6. It also fixes a couple of the most common warnings that were surfacing.

### New Features

* Support Python 3.10 and 3.11 - Issue [#259](https://github.com/sdv-dev/CTGAN/issues/259) by @pvk-developer

### Bugs Fixed

* Fix SettingWithCopyWarning (may be leading to a numerical calculation bug) - Issue [#215](https://github.com/sdv-dev/CTGAN/issues/215) by @amontanez24
* FutureWarning in data_transformer with pandas 1.5.0 - Issue [#246](https://github.com/sdv-dev/CTGAN/issues/246) by @amontanez24

### Maintenance

* CTGAN Package Maintenance Updates - Issue [#257](https://github.com/sdv-dev/CTGAN/issues/257) by @amontanez24

## v0.6.0 - 2022-10-07

This release renames the models in CTGAN. `CTGANSynthesizer` is now called `CTGAN` and `TVAESynthesizer` is now called `TVAE`.

### New Features

* Rename synthesizers - Issue [#243](https://github.com/sdv-dev/CTGAN/issues/243) by @amontanez24

## v0.5.2 - 2022-08-18

This release updates CTGAN to use the latest version of RDT. It also includes performance and robustness updates to the data transformer.

### Issues closed
* Bump rdt version - Issue [#242](https://github.com/sdv-dev/CTGAN/issues/242) by @katxiao
* Single thread data transform is slow for huge table - Issue [#151](https://github.com/sdv-dev/CTGAN/issues/151) by @mfhbree
* Fix RDT api - Issue [#232](https://github.com/sdv-dev/CTGAN/issues/232) by @pvk-developer
* Update macos to use latest version. - Issue [#237](https://github.com/sdv-dev/CTGAN/issues/237) by @pvk-developer
* Update the RDT version to 1.0 - Issue [#224](https://github.com/sdv-dev/CTGAN/issues/224) by @pvk-developer
* Update slack invite link. - Issue [#222](https://github.com/sdv-dev/CTGAN/issues/222) by @pvk-developer
* robustness fix, when data have less rows than the default number of cl… - Issue [#211](https://github.com/sdv-dev/CTGAN/issues/211) by @Deathn0t

## v0.5.1 - 2022-02-25

This release fixes a bug with the decoder instantiation, and also allows users to set a random state for the model
fitting and sampling.

### Issues closed

* Update self.decoder with correct variable name - Issue [#203](https://github.com/sdv-dev/CTGAN/issues/203) by @tejuafonja
* Add random state - Issue [#204](https://github.com/sdv-dev/CTGAN/issues/204) by @katxiao

## v0.5.0 - 2021-11-18

This release adds support for Python 3.9 and updates dependencies to ensure compatibility with the
rest of the SDV ecosystem, and upgrades to the latests [RDT](https://github.com/sdv-dev/RDT/releases/tag/v0.6.1)
release.

### Issues closed

* Add support for Python 3.9 - Issue [#177](https://github.com/sdv-dev/CTGAN/issues/177) by @pvk-developer
* Add pip check to CI workflows - Issue [#174](https://github.com/sdv-dev/CTGAN/issues/174) by @pvk-developer
* Typo in `CTGAN` code - Issue [#158](https://github.com/sdv-dev/CTGAN/issues/158) by @ori-katz100 and @fealho

## v0.4.3 - 2021-07-12

Dependency upgrades to ensure compatibility with the rest of the SDV ecosystem.

## v0.4.2 - 2021-04-27

In this release, the way in which the loss function of the TVAE model was computed has been fixed.
In addition, the default value of the `discriminator_decay` has been changed to a more optimal
value. Also some improvements to the tests were added.

### Issues closed

* `TVAE`: loss function - Issue [#143](https://github.com/sdv-dev/CTGAN/issues/143) by @fealho and @DingfanChen
* Set `discriminator_decay` to `1e-6` - Pull request [#145](https://github.com/sdv-dev/CTGAN/pull/145/) by @fealho
* Adds unit tests - Pull requests [#140](https://github.com/sdv-dev/CTGAN/pull/140) by @fealho

## v0.4.1 - 2021-03-30

This release exposes all the hyperparameters which the user may find useful for both `CTGAN`
and `TVAE`. Also `TVAE` can now be fitted on datasets that are shorter than the batch
size and drops the last batch only if the data size is not divisible by the batch size.

### Issues closed

* `TVAE`: Adapt `batch_size` to data size - Issue [#135](https://github.com/sdv-dev/CTGAN/issues/135) by @fealho and @csala
* `ValueError` from `validate_discre_columns` with `uniqueCombinationConstraint` - Issue [133](https://github.com/sdv-dev/CTGAN/issues/133) by @fealho and @MLjungg

## v0.4.0 - 2021-02-24

Maintenance relese to upgrade dependencies to ensure compatibility with the rest
of the SDV libraries.

Also add a validation on the CTGAN `condition_column` and `condition_value` inputs.

### Improvements

* Validate condition_column and condition_value - Issue [#124](https://github.com/sdv-dev/CTGAN/issues/124) by @fealho

## v0.3.1 - 2021-01-27

### Improvements

* Check discrete_columns valid before fitting - [Issue #35](https://github.com/sdv-dev/CTGAN/issues/35) by @fealho

## Bugs fixed

* ValueError: max() arg is an empty sequence - [Issue #115](https://github.com/sdv-dev/CTGAN/issues/115) by @fealho

## v0.3.0 - 2020-12-18

In this release we add a new TVAE model which was presented in the original CTGAN paper.
It also exposes more hyperparameters and moves epochs and log_frequency from fit to the constructor.

A new verbose argument has been added to optionally disable unnecessary printing, and a new hyperparameter
called `discriminator_steps` has been added to CTGAN to control the number of optimization steps performed
in the discriminator for each generator epoch.

The code has also been reorganized and cleaned up for better readability and interpretability.

Special thanks to @Baukebrenninkmeijer @fealho @leix28 @csala for the contributions!

### Improvements

* Add TVAE - [Issue #111](https://github.com/sdv-dev/CTGAN/issues/111) by @fealho
* Move `log_frequency` to `__init__` - [Issue #102](https://github.com/sdv-dev/CTGAN/issues/102) by @fealho
* Add discriminator steps hyperparameter - [Issue #101](https://github.com/sdv-dev/CTGAN/issues/101) by @Baukebrenninkmeijer
* Code cleanup / Expose hyperparameters - [Issue #59](https://github.com/sdv-dev/CTGAN/issues/59) by @fealho and @leix28
* Publish to conda repo - [Issue #54](https://github.com/sdv-dev/CTGAN/issues/54) by @fealho

### Bugs fixed

* Fixed NaN != NaN counting bug. - [Issue #100](https://github.com/sdv-dev/CTGAN/issues/100) by @fealho
* Update dependencies and testing - [Issue #90](https://github.com/sdv-dev/CTGAN/issues/90) by @csala

## v0.2.2 - 2020-11-13

In this release we introduce several minor improvements to make CTGAN more versatile and
propertly support new types of data, such as categorical NaN values, as well as conditional
sampling and features to save and load models.

Additionally, the dependency ranges and python versions have been updated to support up
to date runtimes.

Many thanks @fealho @leix28 @csala @oregonpillow and @lurosenb for working on making this release possible!

### Improvements

* Drop Python 3.5 support - [Issue #79](https://github.com/sdv-dev/CTGAN/issues/79) by @fealho
* Support NaN values in categorical variables - [Issue #78](https://github.com/sdv-dev/CTGAN/issues/78) by @fealho
* Sample synthetic data conditioning on a discrete column - [Issue #69](https://github.com/sdv-dev/CTGAN/issues/69) by @leix28
* Support recent versions of pandas - [Issue #57](https://github.com/sdv-dev/CTGAN/issues/57) by @csala
* Easy solution for restoring original dtypes - [Issue #26](https://github.com/sdv-dev/CTGAN/issues/26) by @oregonpillow

### Bugs fixed

* Loss to nan - [Issue #73](https://github.com/sdv-dev/CTGAN/issues/73) by @fealho
* Swapped the sklearn utils testing import statement - [Issue #53](https://github.com/sdv-dev/CTGAN/issues/53) by @lurosenb

## v0.2.1 - 2020-01-27

Minor version including changes to ensure the logs are properly printed and
the option to disable the log transformation to the discrete column frequencies.

Special thanks to @kevinykuo for the contributions!

### Issues Resolved:

* Option to sample from true data frequency instead of logged frequency - [Issue #16](https://github.com/sdv-dev/CTGAN/issues/16) by @kevinykuo
* Flush stdout buffer for epoch updates - [Issue #14](https://github.com/sdv-dev/CTGAN/issues/14) by @kevinykuo

## v0.2.0 - 2019-12-18

Reorganization of the project structure with a new Python API, new Command Line Interface
and increased data format support.

### Issues Resolved:

* Reorganize the project structure - [Issue #10](https://github.com/sdv-dev/CTGAN/issues/10) by @csala
* Move epochs to the fit method - [Issue #5](https://github.com/sdv-dev/CTGAN/issues/5) by @csala

## v0.1.0 - 2019-11-07

First Release - NeurIPS 2019 Version.

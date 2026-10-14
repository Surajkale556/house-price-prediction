Challenges log - PRCP-1020 House Price Prediction

#===========================================================
Entry 1: pandas default loading erases the category/missing distinction

problem: 

the dataset uses text as (NA/None) to mean the house does not have this feature
a default pd.read_csv convert this values as NaN
making house having no pool indestinguishable from "value unknown" 
if we impute that column some houses which doesnt have a pool will show that they have it making data corrupt.

Tried: pd.read_csv(..., keep_default_na=False, na_values=[""])

Why: keep_default_na=False  disables the function so "NA/None" load as ordinary categories
na_values=[""] keeps genuinely empty cells as NaN so true missingness stays countable

Result: Distinction preserved. Verified: MasVnrType shows "None" (864) as acategory; Alley shows "NA" (1369) as a category.

#===============================================================

Entry 2 - the provided file stored ALL blanks as "NA" strings

problem: df.isna().sum() showed 0 missing values
feature absence and value missing was hard to diffrentiate
3 column were misclassified as str data type due to presence of "NA"

Tried: Data type audit - compared all 81 columns against their documentated types 
dignosed each broken column against related features to get their values understanding before making any changes
(GarageYrBlt "NA" rows vsGarageType; MasVnrArea "NA" rows vs MasVnrType; LotFrontage "NA" rows vsNeighborhood). Only then repaired with pd.to_numeric(errors="coerce").

why: "NA" means different things per column : absence or genuine missing. Repairing without diagnosinsing would have mixed the meanings

Result: 38 numeric columns verfied = 35 int64 + 3 float64
True Nan VISIBLE =LotFrontage 259, GarageYrBlt 81 (absence),MasVnrArea 8.

#==============================================================

Entry 3 - documentation vs file column name mismatch 

problem: Spec says "Bedroom" and "Kitchen"; file says "BedroomAbvGr" and"KitchenAbvGr". File also has an "Id" column absent from the documented list.

tried: Cross-checked documented names against df.columns.

Why: can cause wrong featureselection later

Result: 3 differences found; other 78 names match. Noted in audit findings;all code uses actual file names.

#==============================================================

Entry 4 - two houses where "NA" is genuine missing not absent in basement related column
ALL 19 columns missing value and nonexistant value classified with reasoning all table data is clear


problem: the five basement column should agree but the count were 37/37/38/38/38
BsmtExposure and BsmtFinType2 eachcarried one extra "NA".

Tried: took no basement as TotalBsmtSF == 0
and then searched for mismatch betweeen each column and does the basement exist or not

why: "NA" in this dataset can be missing value or nonexistant value
since basement area has a value we cant treat corresponding basement attribute as non existant it must have a value

result: id 949 and id 333 both classified as genuine missing and need to be imputed

ALL 19 columns missing value and nonexistant value classified with reasoning all table data is clear

#========================================================

Entry 5: MasVnrType hold both a valid category and genuine missing

problem - "None"(864)- documented value has meaning 
           and "NA"(8)- is not documented genuine missing
           both live in same column and need different handling

outcome: "None" kept as is
        "NA " is classified genuine missing

#========================================================

Entry 6 : "NA" HAS DIFFERENT MEANING IN DIFFERENT COLUMNS
        (genuine missing, absense, validcategory)

problem:   categorizing the meaning of NA and deciding the action on it


tried:   using logic and cross comparing
FireplaceQu  = all 690 FireplaceQu "NA" houses have Fireplaces = 0
GarageType  = all 81 GarageType "NA" houses have GarageCars = 0 and GarageArea = 0
BsmtQual  = all 37 BsmtQual "NA" houses have TotalBsmtSF = 0
reverse: all 7 PoolArea > 0 houses carry a PoolQC rating; zero houseswith fireplaces lack FireplaceQu

Outcome: 
    (19 columns; other 62 completely clean)
    decided which value need which type of fill
     Stated limitation: Alley, Fence, MiscFeaturehave no numeric companion column — trusted on documentation alone.

#==========================================================

Entry 7: GarageYrBlt : absence inside a year column

problem:
    81 NaN = houses with no garage. Filling 0 creates a meaningless"year 0" that breaks any age-based feature.

solution:
    Fill with YearBuilt + add a HasGarage binary flag.

Reason: putting it as year 0 makes the data inconsistant.

#=============================================================

Entry 8: LotFrontage : 259 genuine missing

problem : missing values without a pattern 
        if tried to impute based on whole data the chances of data leak in training data is high

solution: impute inside the sklearn pipeline.

#============================================================
Entry 9: skewness

Problem: SalePrice is right-skewed(1.88)- the expansive houses dominates squared error

Technique: Model log1p(SalePrice) 

Why: log 1p is zero-safe and it turns percentage relationships additive- the shape most linear model assume

Result: skew 1.88 -> 0.12

#===============================================================
Entry 10: two contradictory houses

Problem: ids 529 and 1299 each value is plausible alone
but jointly absurd, they teach model that hugh area means low price

Technique: Scatterplot + SaleCondition cross-check

Why: Both are partial sales incomplete transactions whose price doesnt reflect completed market value
ids 692 and 1183 are eqaully large but have corresponding larger price

Result: ID 524 AND ID 11299 flagged for removal

#============================================================
Entry 11: yearRemodAdd 1950 clamp

Problem: spike at exactly 1950 

Technique:
treat 1950 as unknown/default not a geniune remodel year

why: "years since remodel" would credit those house with renovation they may never had

Result: Noted for future action

#=========================================================
Entry 12: Raw ANOVA F misled on many-category column

Problem:
F- ranking placed neighbourhood at #12 despite location being a primary driver. it penalize having bigger number of categories

Technique:
Re-ranked with η² 

Why:
η² is category-count-fair; F inflates few-category columns 

Result: Neighborhood 0.57 → #2, OverallQual 0.67 → #1. Also exposed MSSubClass:−0.08 as a number vs η² 0.33 as a category — it must be recast.

#===========================================================
Entry 13:IQR rule degenerates on zero-inflated column

Problem : sweep flagged 208 outliers in EnclosedPorch(and 8 more columns)

Technique: Diagnosed the rule: >75% zeroes -> q1=q3=0
-> iqr=0 every non zero value flagged

Why:
the rule silently break on data shape:
flag counts are feature haverinventories not anomalies

Resutl:
Zero data errors: zero heavy clumn becomes binary flags 
only 524/1299 removed


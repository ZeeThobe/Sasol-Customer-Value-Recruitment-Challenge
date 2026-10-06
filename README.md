# Sasol Customer Value Recruitment Challenge

Solution workspace for the [Sasol Customer Value Recruitment Challenge](https://zindi.world/competitions/sasol-customer-value-recruitment-challenge) hosted on Zindi.

> **The business question:** *Given everything we know about a customer from their past Sasol transactions, what are they likely to do in the next three months, and what value or opportunity could that create?*

This is a **recruitment challenge**: Sasol uses it to find data-science talent. Leaderboard rank gets you shortlisted. After that, top candidates are typically asked for their CV, their reproducible code and an explanation of their solution, and some are invited to interview. Clean, well-documented, reproducible work counts as much as the score.

---

## 1. Task overview

Sasol Rewards loyalty transactions at Sasol forecourts and convenience stores cover **19 March 2024 to 30 November 2025**. For each test customer, predict what happens in the **three calendar months after the cutoff**, which is **1 Dec 2025 to 28 Feb 2026**.

| Target | Type | What it measures |
|---|---|---|
| `CLV_fuel` | Regression | Normalised `ln(1 + fuel litres)` bought in the outcome window |
| `CLV_nonfuel` | Regression | Normalised `ln(1 + non-fuel rand spend, excl. VAT)` in the outcome window |
| `Opportunity` | Multiclass (17 classes) | The customer's single most valuable opportunity: category growth, new-category adoption, Stable or Inactivity |

The submission has one row per customer and a single combined `Opportunity` label. There are no separate opportunity or category columns.

## 2. Data

| File | Contents |
|---|---|
| `train.csv` | Line-level historical loyalty transactions (the full released history) |
| `test.csv` | Customers who need predictions at the 1 Dec 2025 cutoff |
| `SampleSubmission.csv` | Required submission layout, with exact column names and order |

Approximate size of the release: **467,880 line items**, **365,638 baskets** (`TransactionId`), **6,204 customers** in history, **5,488 customers** in the test file, and **390 locations**.

### Data dictionary

| Column | Meaning |
|---|---|
| `DateTimeZA` | Transaction timestamp in South African local time |
| `TransactionId` | Basket identifier. Several rows can share one basket |
| `LineItemSequence` | Line order within the basket |
| `LocationId` | Store or site identifier |
| `ID` | Anonymised customer identifier. **Read it as a string** |
| `LineItemType` | Line type. Fuel regression uses `Sale_Fuel` |
| `TransactionTotalInclAmount` | Whole-basket total, **repeated on every line**. Never sum it over lines |
| `StockItemId` | Product identifier |
| `ItemName` | Product or adjustment name |
| `ItemCategoryLevel1` | Coarse category |
| `ItemCategoryLevel2` | **Category used by the target rules.** The literal value `NA` is a real category, not a missing value |
| `ItemCategoryLevel3` | Finer category |
| `Quantity` | Signed quantity. Litres for `Sale_Fuel`, source units otherwise |
| `TotalExclAmount` | Signed line amount. Store items are pre-tax, but **fuel amounts include tax**. Used directly by the label rules |
| `TotalModifierAmount` | Price modifier, already applied. Do not subtract it again |
| `UnitSellingExclPrice` | Unit price. Do not rebuild amounts from price × quantity − modifiers |

Pitfalls when loading the data:
- Use `pd.read_csv(..., dtype=str, keep_default_na=False)` so that `ID` stays a string and the `NA` category doesn't turn into NaN.
- Returns are negative rows and reduce net totals.
- Count purchases as distinct baskets, not as lines.

## 3. How the targets are built (official label rules)

The organisers provided a reference label generator (`label_rules.py` + `label_config.json`). Its rules are summarised here.

**Eligibility.** At a given cutoff, a customer is eligible if they have **at least 3 distinct `TransactionId`s strictly before the cutoff**. Every transaction type counts. Future data is never used to decide eligibility.

**Windows.**
- The **outcome** is the next 3 calendar months.
- The **growth comparison** is the previous 3 calendar months.
- **Adoption** checks *all* history before the cutoff.

**Qualifying purchase.** A line with `Quantity > 0` **and** `TotalExclAmount > 0`. Count distinct baskets.

**Ignored rows.** The adjustment items `TOTAL/ROUNDING DISCO`, `PENSIONER DISCOUNT` and `WOW_PROMO` never count toward targets or activity.

**Excluded categories.** `Value added services`, `Print media` and `Carrier Bags` count as *no activity*. `NA` and `Unknown` count as real activity: they add to non-fuel spend and keep a customer out of Inactivity.

### Regression targets

- **Fuel:** the sum of signed `Quantity` (litres) where `ItemCategoryLevel2 == "Fuel"` and `LineItemType == "Sale_Fuel"`.
- **Non-fuel:** the sum of signed `TotalExclAmount` over every category except Fuel and the three excluded categories.
- **Normalisation:** `z = ln(1 + y) / s`, not centred, using fixed public constants:

| Target | `s` |
|---|---|
| `CLV_fuel` | `2.352825386185113` |
| `CLV_nonfuel` | `2.4200904988927476` |

`s` is the population standard deviation of `ln(1+y)` over five quarterly calibration snapshots (cutoffs 2024‑09‑01, 2024‑12‑01, 2025‑03‑01, 2025‑06‑01, 2025‑09‑01). Do **not** recompute it.

### Opportunity label

Each `ItemCategoryLevel2` category (excluding the three non-activity categories) is evaluated on its own:

- **Existing-category growth** needs all of the following:
  - a historical qualifying purchase in the category
  - at least 1 purchase in the outcome window
  - a net spend increase of **at least R20** compared with the previous quarter
  - future spend **more than 25% higher** than previous-quarter spend
- **New-category adoption** needs no prior qualifying purchase in the category, at least 1 purchase in the outcome window, and positive future net spend.

**Choosing the winner.** Candidates are compared on rand value. The growth increment is future spend minus previous-quarter spend. The adoption increment is future spend. Fuel is compared using `TotalExclAmount`, which includes tax, not litres.
- The highest increment wins.
- On a tie, adoption beats growth, then the alphabetically first category wins.
- Categories compete individually **before** being mapped to `Other`. Spend is never pooled across categories that share the `Other` label.

**Fallback labels.**
- **Inactivity** means there were no qualifying in-scope purchases in the outcome window. A customer whose only activity was returns is also Inactivity.
- **Stable** means the customer was active but no category qualified for growth or adoption. Declining customers can be Stable.

**The 17 valid classes** (spelling is authoritative):

```
Existing-category growth: Beverages
Existing-category growth: Food & beverage
Existing-category growth: Fuel
Existing-category growth: Other
Inactivity
New category adoption: Beverages
New category adoption: Braai & Ice
New category adoption: Breads & rolls
New category adoption: Confectionary
New category adoption: Food & beverage
New category adoption: Groceries
New category adoption: Ice-cream
New category adoption: Lubricants
New category adoption: Other
New category adoption: Snacking
New category adoption: Tobacco
Stable
```

Category mapping:
- **Growth labels:** only Beverages, Food & beverage and Fuel keep their own name. Every other category maps to `Other`.
- **Adoption labels:** Beverages, Braai & Ice, Breads & rolls, Confectionary, Food & beverage, Groceries, Ice-cream, Lubricants, Snacking and Tobacco keep their own name. Everything else maps to `Other`, including Fuel, Coffee, Car Wash, Krispy Kreme, NA and Unknown.

## 4. Evaluation

| Component | Metric | Weight |
|---|---|---|
| `CLV_fuel` | RMSE | 0.3 |
| `CLV_nonfuel` | RMSE | 0.3 |
| `Opportunity` | Weighted F1 | 0.4 |

## 5. Rules and logistics

- **Closes:** 19 October 2026
- **Entry:** solo only. Do not share code privately with other participants
- **Data:** only the supplied datasets. No external commercial, demographic, geospatial or economic data
- **Tools:** open-source tools only. **No AutoML**
- **Reproducibility:** fix random seeds (seed 42)
- **Submissions:** at most 10 per day and 200 in total. Choose **2 final submissions** for private-leaderboard scoring
- **Code:** keep the solution repository **private** while the competition is running

## 6. Validation strategy (suggested)

The label generator can be run at any earlier cutoff whose full 3-month outcome window lies inside the released history. That lets you build leakage-safe training snapshots:

| Snapshot cutoff | Features from | Labels from |
|---|---|---|
| 2024‑09‑01 … 2025‑09‑01 (quarterly), or monthly cutoffs | history `< cutoff` | `[cutoff, cutoff + 3 months)` |
| **2025‑12‑01 (test)** | history `< 2025‑12‑01` | hidden (Dec 2025 – Feb 2026) |

The last fully observable cutoff is 2025‑09‑01, so use it as the hold-out fold. Train on earlier cutoffs and report RMSE and weighted F1 with the same weighting as the leaderboard.

---

*Some details here come from public documentation and competitor repositories, not directly from the Zindi page. Before relying on them, check the competition's **Info**, **Data** and **Rules** tabs on Zindi, especially the metric formula, column names and prize or recruitment terms.*

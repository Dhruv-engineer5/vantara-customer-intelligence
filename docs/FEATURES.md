# Feature dictionary

Every model input, with how it is calculated and why it should carry predictive signal. All features use only invoice rows
strictly before the cutoff date (2011-09-09), so none of them can see the churn window. Statistics that are learned from data
(product median prices, StockCode frequencies, engagement percentile reference) are fitted on **training customers only**,
saved in `models_artifacts/reference_stats.joblib` and reused unchanged by the API. A unit test checks that this file lists
every column the model uses.

| Feature | Calculation | Why it helps (one line) |
|---|---|---|
| `recency_days` | Days from the customer's last purchase to the cutoff | Customers who bought recently are more likely to buy again; the strongest classic churn signal. |
| `orders_last_30d` | Orders in the 30 days before the cutoff | Very recent activity shows current engagement. |
| `orders_last_90d` | Orders in the 90 days before the cutoff | Same window length as the churn horizon, so it reflects the recent run-rate. |
| `orders_last_180d` | Orders in the 180 days before the cutoff | Medium-term activity; separates steady buyers from one-off bursts. |
| `spend_last_30d` | Net spend in the last 30 days | Recent money flow; falling spend often precedes churn. |
| `spend_last_90d` | Net spend in the last 90 days | Recent value at the churn-horizon scale. |
| `spend_last_180d` | Net spend in the last 180 days | Medium-term value; one of the top SHAP features. |
| `recent_share_90d` | Orders in the last 90 days divided by all orders | Shows whether activity is accelerating or fading against the customer's own history. |
| `days_since_first_purchase` | Days from the first purchase to the cutoff | Customer age in the data; newer customers behave differently from long-standing ones. |
| `frequency` | Number of distinct invoices before the cutoff | How habitual the customer is; the F in RFM. |
| `monetary_total` | Total gross positive sales (returns excluded) before the cutoff | Overall customer value; the M in RFM. |
| `monetary_avg` | Average gross sales per invoice (returns excluded) | Order size; separates large-basket from small-basket buyers. |
| `tenure_days` | Days between first and last purchase | Length of the active relationship. |
| `avg_basket_size` | Average number of items per invoice | Wholesale-like buyers behave differently from retail buyers. |
| `avg_lines_per_order` | Average number of invoice lines per order | Breadth of each order. |
| `n_unique_products` | Distinct StockCodes bought | Product breadth; broad buyers have more reasons to return. |
| `gap_mean_days` | Mean days between consecutive orders | The customer's own buying rhythm; a long silence against the usual gap signals churn. |
| `gap_var_days` | Variance of days between orders | Regular buyers are easier to predict than erratic ones. |
| `freq_trend` | Slope of monthly order counts over the 12 months before the cutoff | Rising or falling purchase frequency. |
| `seasonal_concentration` | Share of spend in the customer's biggest calendar month | Holiday-only buyers churn after the season. |
| `aff_christmas_seasonal` | Share of the customer's items in the keyword category 'christmas_seasonal' | Product-category taste; feeds the recommendation and next-category logic. |
| `aff_kitchen_dining` | Share of the customer's items in the keyword category 'kitchen_dining' | Product-category taste; feeds the recommendation and next-category logic. |
| `aff_bags_storage` | Share of the customer's items in the keyword category 'bags_storage' | Product-category taste; feeds the recommendation and next-category logic. |
| `aff_lights_candles` | Share of the customer's items in the keyword category 'lights_candles' | Product-category taste; feeds the recommendation and next-category logic. |
| `aff_home_decor` | Share of the customer's items in the keyword category 'home_decor' | Product-category taste; feeds the recommendation and next-category logic. |
| `aff_toys_party` | Share of the customer's items in the keyword category 'toys_party' | Product-category taste; feeds the recommendation and next-category logic. |
| `aff_garden_outdoor` | Share of the customer's items in the keyword category 'garden_outdoor' | Product-category taste; feeds the recommendation and next-category logic. |
| `aff_other` | Share of the customer's items in the keyword category 'other' | Product-category taste; feeds the recommendation and next-category logic. |
| `return_rate` | Share of the customer's lines that are returns | Dissatisfaction or fit problems. |
| `discount_sensitivity` | Share of lines priced below 90% of the product's training median price | Customers who buy mainly at discount react to promotions; median prices are fitted on training customers. |
| `product_popularity_mean` | Mean frequency encoding of the customer's StockCodes (log of 1 + number of training customers who bought each product; unseen product = 0) | Mainstream versus specialist taste. This is the brief's frequency encoding of StockCode-derived features, fitted on training customers only and persisted in reference_stats.joblib. |
| `niche_product_share` | Share of lines whose product was bought by fewer than 5 training customers | Specialist buyers have different retention behaviour. |
| `country_united_kingdom` | 1 if the customer's main country is 'united kingdom' | One-hot encoding for the low-cardinality Country field; country mix differs in service and delivery. |
| `country_eire` | 1 if the customer's main country is 'eire' | One-hot encoding for the low-cardinality Country field; country mix differs in service and delivery. |
| `country_germany` | 1 if the customer's main country is 'germany' | One-hot encoding for the low-cardinality Country field; country mix differs in service and delivery. |
| `country_france` | 1 if the customer's main country is 'france' | One-hot encoding for the low-cardinality Country field; country mix differs in service and delivery. |
| `country_other` | 1 if the customer's main country is not one of the listed groups | One-hot encoding for the low-cardinality Country field; the remaining groups are separate columns. |
| `clv_historical` | Net revenue (sales minus returns) over the observed history | Past value, also reported next to predicted value. |
| `engagement_score` | Mean training-reference percentile of inverted recency, frequency and monetary value, scaled to 0-100 | One-number health score used by the dashboard. |

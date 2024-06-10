# Market Basket Analysis

By using `Association rules` derived from `frequent itemsets` identified by `Apriori`, Market Basket Analysis helps retailers discover hidden patterns in customer buying habits, leading to better product placement, targeted promotions, and improved inventory management. Check out this **[Kaggle notebook](https://www.kaggle.com/code/wilfridawere/market-basket-analysis)** for the full code.

I will use the **[Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)** from UCI machine learning repository

These are the steps I will follow:

1. Number of Transactions by Country
2. Working with Transactions from France
3. Binary Encoding
4. Training the Model
5. Making Recommendations

***I'll briefly explain Association rules and the Apriori algorithm. If you're already familiar with these concepts, you can skip ahead.***

## Association Rules:

Imagine a grocery store. An association rule might state: "If a customer buys bread, then they are also likely to buy butter." This rule captures the relationship between frequently bought together items.

Association rules are written in the format A -> B, where A (antecedent) is the item purchased first and B (consequent) is the item purchased together with A.

## Apriori:

The Apriori algorithm efficiently finds frequently bought together items (itemsets) in your transaction data. `mlxtend` library provides a user-friendly implementation. You simply:

* Set Minimum Support: Define the minimum number of times an itemset needs to appear to be considered frequent (e.g., 10% of transactions).
* Run Apriori: Use `mlxtend.frequent_patterns.apriori` on your transaction data.
* Get Frequent Itemsets: Apriori identifies groups of items (bread & butter, milk & cereal) that frequently appear together, exceeding the minimum support threshold.

## Interpreting the results 

Choosing the Right Metric:

1. `Support`: A good starting point to identify frequently bought-together itemsets.
2. `Confidence`: Helps refine the analysis by focusing on how likely B is purchased with A.
3. `Lift`: Measures how much more likely it is to buy B given A compared to buying them independently. A higher lift indicates a stronger association.

In essence:

1. Use higher support to identify commonly purchased itemsets.
2. Use higher confidence to identify strong conditional relationships between items.
3. Use higher lift to identify itemsets where co-purchases are more frequent than expected by chance.

# Data Cleaning

The dataset has 541,909 rows and 8 columns

'Description' column has 0.3% missing values so we can drop them.

# 1. Number of Transactions by Country

Check out the full output on **[Kaggle](https://www.kaggle.com/code/wilfridawere/market-basket-analysis)**

| Country | Transactions |
|---|---|
| United Kingdom | 486,168 |
| Germany | 9,042 |
| France | 8,408 |
| EIRE | 7,894 |
| Spain | 2,485 |
| Netherlands | 2,363 |
| Belgium | 2,031 |
| Switzerland | 1,967 |
| Portugal | 1,501 |
| Australia | 1,185 |

# 2. Working with Transactions from France

```python
# Filter dataframe and select transactions where 'Country' is 'France'
# Groups transactions based on 'InvoiceNo', then by 'Description'
# Within each group, calculate the total 'Quantity' of each product (identified by description) purchased in each 'InvoiceNo'.
france_basket = df[df['Country'] =='France'] \
               .groupby(['InvoiceNo','Description'])['Quantity'] \
               .sum().unstack().reset_index().fillna(0) \
               .set_index('InvoiceNo')

print(france_basket.shape)  # (392 rows and 1564 columns)
print() # blank line
france_basket.head()
```
The summarised output looks as follows:

![Alt text](./)

# 3. Binary Encoding

***Focus on Presence/Absence***: The function encodes positive values (indicating a purchase) to True, essentially signifying the item's presence in the transaction.

***Handling Non-Positive Values***: It encodes everything less than or equal to 0 (including 0) to False, representing the item's absence in the transaction.

Focusing on presence/absence simplifies market basket analysis by focusing on co-occurrence patterns, not quantities.

```python
def my_encode_units(x):
  """Converts positive values to True and everything else to False."""
  return x > 0  # This returns True for positive values and False otherwise

my_basket_sets = france_basket.map(my_encode_units)
```

## Top 20 Most Frequent Products in France

![Alt text](./)

# 4. Training the Model

Adjust the min_support value if you want to. You will see different results.

`min_support=0.05`: This parameter sets the minimum support threshold. Here, it means an itemset needs to appear in at least 5% of transactions to be considered frequent. I generate **86 rules**

A higher support value like 10% gives less rules while a lower one like 1% gives more than 6,000 rules! 

This is how to ***Train the Apriori model and generate association rules***:

```python
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Generate frequent itemsets
my_frequent_itemsets = apriori(my_basket_sets, min_support=0.05, use_colnames=True)

# Generate rules
my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=1)

print(my_rules.shape) # 86 rows so 86 association rules

# View first 10 rules
my_rules.head(10)
```
So these are the first 5 rules

![Alt text](./)

# 5. Making Recommendations
​
You can sort the rules in descending order by `Lift` and make relevant recommendations. Also consider the `support` and `confidence` values.

![Alt text](./)

## Strong Co-occurrence of Party Supplies

Looking at the first two rules:

* Both rules involve "PACK OF 6 SKULL PAPER CUPS" and "PACK OF 6 SKULL PAPER PLATES" with the same support value (0.051020) and lift value (14.254545)
* `Lift: Highlighting a Non-Random Relationship`
* A lift value greater than 1 indicates a positive association, but a value this high (14.25) suggests a very **strong, non-random relationship** between the two products.

These rules highlight a strong co-occurrence pattern between the two skull-themed party supplies, indicating that customers who buy one item are very likely to also buy the other.  Here's how a **Store manager can leverage this information**:

1. **Strategic Product Placement**: Consider recommending "PACK OF 6 SKULL PAPER PLATES" next to "PACK OF 6 SKULL PAPER CUPS". This close proximity can create a visual reminder for customers browsing either product, potentially leading to impulse purchases of the complementary item (cups if looking at plates, and vice versa).

2.  **Ensure Adequate Stock**: Given the high confidence values (especially the 90.9% confidence for plates leading to cups), it's crucial to ensure the store has sufficient stock of both products to meet this customer demand. Running out of either product could lead to lost sales opportunities.

## Product-Specific Recommendations

This section is relevant if you want to see rules on **Specific products** and make Recommendations based on the `support`, `confidence` and `lift` values.

There are five outputs but we will focus on the one with a higher Lift

![Alt text](./)

## Strong Association for "ALARM CLOCK BAKELIKE RED"

The rule suggests a strong association between buying "ALARM CLOCK BAKELIKE GREEN" or "ALARM CLOCK BAKELIKE PINK" (antecedents) and also buying "ALARM CLOCK BAKELIKE RED" (consequent)

1. **Support** (0.0638): This indicates that out of all transactions, approximately 6.4% involved customers buying at least one of the green or pink alarm clocks along with the red alarm clock.

1. **Confidence** (0.862): If a customer buys "ALARM CLOCK BAKELIKE GREEN" or "ALARM CLOCK BAKELIKE PINK", there's an 86.2% chance they also purchase the "ALARM CLOCK BAKELIKE RED". This suggests a strong tendency for customers who like these similar alarm clocks to also consider the red one.

1. **Lift** (9.13): This is the most crucial metric. A lift value this high (over 9) signifies that customers who buy the green or pink clocks are 9.13 times more likely to also buy the red clock compared to if they hadn't purchased the green or pink ones. This indicates a very strong, non-random association.

Recommendation:

* **Cross-Selling**: The store manager should focus on cross-selling the "ALARM CLOCK BAKELIKE RED" whenever a customer shows interest in or purchases either the green or pink version.

* **Visual Merchandising**: Consider creating an eye-catching display that showcases all three alarm clock colors together. This can further emphasize the color options and potentially encourage indecisive customers to buy more than one.

* **Inventory Management**: Ensure there's adequate stock of all three colored alarm clocks, especially the red one, given the high likelihood of customers buying it alongside the green or pink versions.
3.  **Bundled Promotions**: While the co-occurrence is strong, you might consider a bundled promotion (e.g., "Party Pack: Skull Paper Cups & Plates") to further incentivize customers to buy both items together. However, analyse your profit margins before implementing this strategy.

Feel free to perform further datetime-specific analyses and transactions in other Countries.

***Explore and be teachable***

# Hotel Booking – Price Elasticity Modeling & Customer Segmentation

## Project Overview

This project explores **how guest booking behavior responds to price changes**, while identifying **distinct demand patterns** across customer segments. Rather than optimising price, the analysis prioritises understanding *demand elasticity*, customer behavior, and acquisition channel dynamics. The core methodology blends **price elasticity estimation**, **demand-driven segmentation**, and **strategy recommendations**, built within a single, reproducible notebook.

**NOTE**: This project is part of my weekly series in efforts to **demystify applied statistical techniques through real-world, project-driven examples**, making concepts like causal inference, elasticity modeling, and behavioral segmentation more accessible to practitioners of all backgrounds.

### Author

**Einstein Ebereonwu** • [GitHub](https://github.com/munas-git) • [LinkedIn](https://www.linkedin.com/in/einstein-ebereonwu/)   
*Dataset: [Kaggle – Hotel Booking](https://www.kaggle.com/datasets/ahmedwaelnasef/hotel-booking)*

---

## Pipeline Highlights

| Stage                         | Key Activities                                                                                                                                                                                 |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Quality Review**      | • Inspected distribution stats to identify implausible values (e.g. zero room price, >8 children).<br>• Detected anomalies using **Isolation Forests** (outliers ≈ 0.1%).                        |
| **Outlier Handling**         | • Outliers removed based on `iso_score`.<br>• All `avg_room_price > 400` were also excluded to avoid skew.                                                 |
| **Missing & Invalid Values** | • Replaced `avg_room_price = 0` using a **KNN regressor** trained on clean features.<br>• Detected & removed bookings recorded as `2018-02-29` (non-leap year).                                |
| **Feature Engineering**      | • Label/Ordinal encoded: `meal_plan`, `room_type`, `status`.<br>• One-hot encoded `market_segment`.<br>• Created `booking_date` from year, month, day.                                          |
| **Target Variable Design**   | • Defined *demand* as `number of non-cancelled bookings per day`.                                                                                          |
| **Exploratory Data Analysis**| • Seasonal patterns and demand spikes uncovered.<br>• `Room Type 1` and `Room Type 4` emerged as most consistently booked.<br>• Clear seasonal peaks in August–October.                          |
| **ANOVA & Correlation**      | • Identified key demand drivers: `room_type`, `lead_time`, `avg_room_price`, `market_segment_Complementary`.                                               |
| **Elasticity Modeling**      | • Modeled **price elasticity** with OLS (linear + polynomial).<br>• Demand for Room 1 is **mostly inelastic**, turning elastic beyond ~$120.               |
| **Controlled Regression**    | • Added controls: `lead_time`, `weekend_nights`, `month`, and `market_segment`.<br>• Captured a **bimodal** elasticity distribution hinting multiple behavioral groups.                         |
| **Demand Segmentation**      | • Segmented Room 1 guests into **3 peak groups** using kernel density clustering.<br>• Compared groupwise behavior (lead time, price, cancellations, market segment).                          |
| **Interaction Modeling**     | • Included interaction terms (e.g. `log_price × lead_time`, `log_price × online`) to better explain variance in elasticity across groups.                  |

---

### Tech Stack

Python | pandas | scikit-learn | statsmodels | seaborn/matplotlib | Isolation Forest | OLS Regression | KMeans/KDE clustering

---

# Key Findings

## **Room Type 1**
- **`Cluster 0` - Likely Leisure Travellers**
    - These customers appear to be **slightly price sensitive** and tend to **compare options** carefully before booking. They are **value seekers** who go for the **best deals**.
    - They are **organised planners**, as they account for the **highest frequency of custom requests**, and **longest lead times (60+ days)**.
    - They also to tend **stay the longest** at the hotel on average. Eventhough they are **not the most frequent guests**, they are **commitment** to their bookings, with **low cancellation rates** reflect their decision-making certainty.
    - They book through both offline and online channels, though like most other guests, the **online channel is slightly more favored**.

- **`Cluster 1` - More Flexible Customers**
    - These guests are **more flexible** in their planning, they do **not plan far in advance**, compared to Cluster 0.
    - They tend to **stay longer** than leisure travelers, especially during **week nights**, and their **spending is average**, neither high or low... However, they are less price sensitive, and are ***able to pay extra*** for perceived fairness.

- **`Cluster 2` - Premium Customers (Likely Business Travellers)**
    - These customers **highly value flexibility**, they have the **highest cancellation rates**, **shortest booking lead times**, and **shorter stays**, mostly on **weekdays**.
    - This behavior suggests **business travel**, likely for events or meetings in the hotel’s surrounding area. They are **price inelastic**, willing to **spend significantly** for convenience or necessity, and their primary driver seems to be **location, flexibility and timing**, rather than price or leisure.


### **Cluster Recommendations**

- **`Cluster 0` - Leisure Travellers**
    - Offer **targeted promotions** such as early bird discounts, **tourist attraction arrangements** or bundle deals.
    - Encourage **direct bookings** through loyalty incentives and price guarantees... This should encourage them to return as guests, should they decide to visit the city again.
    - Highlight **customisable options** (room upgrades, flexible check-in) to appeal to their planning behavior.

- **`Cluster 1` - Flexible Guests**
    - Promote **mid-range packages** and **extended-stay discounts**.
    - Experiment with **last-minute deals** to encourage shorter planning cycles.
    - Market **weekday offers** to align with their preference for weeknight stays.

- **`Cluster 2` - Premium/Business Travellers**
    - Provide **corporate or executive packages** that offer high flexibility.
    - Emphasise **proximity to business or event venues**, fast check-in/out, and premium amenities.
    - No generic discounts whatsoever, focus on **value-added services** like airport pickup, express laundry...

## **Room Type 4**

The **KDE informed KMeans clustering results for Room Type 4** reveal **two distinct traveler segments** whose behaviors **differ significantly from those in Room Type 1**. Unlike the mixed and moderately overlapping patterns seen earlier, these clusters display **sharply contrasting characteristics**, especially in terms of **commitment to bookings**, **price sensitivity**, and **planning behavior**.


- **`Cluster 0` - Budget-Conscious, Uncertain Bookers (High Cancellation Risk)**

    - **Price-Sensitive**: These guests are generally **conscious of cost** and show **tendencies to compare options** before committing.
    - **High Cancellation Risk**: They exhibit **frequent cancellation behavior**, indicating **indecision or shifting plans**.
    - **Moderate Planning Behavior**: Their **booking timeframe** suggests some level of planning, but not necessarily strong commitment.
    - **Diverse Booking Channels**: These guests use a mix of **online and offline platforms**, likely to explore and compare offers.
    - **Lower Interest in Customisation**: Tend to make **fewer special requests**, indicating a focus on basic needs rather than tailored experiences.
    - **Lower Demand Volume**: They generate **fewer bookings overall**, reflecting either lower loyalty or less frequent travel.

- **`Cluster 1` - Committed, High-Spending Guests (Almost Inelastic)**

    - **High Willingness to Pay**: These guests are ready to **spend more** for comfort, convenience, or perceived value, showing **low price sensitivity**.
    - **Very Low Cancellation Risk**: They are **reliable and committed**, with **little to no history of canceling bookings**.
    - **Strong Planning Habits**: Bookings are typically made **well in advance**, suggesting deliberate decision-making and **clear intent** to follow through.
    - **High Customisation Preference**: These guests frequently request **personalised services or specific arrangements**, showing high engagement.
    - **Digitally Engaged**: They **much prefer online booking channels**, reflecting familiarity and comfort with digital tools.

### **Cluster Recommendations**

- **`Cluster 0` - Budget-Conscious, High Cancellation Risk**
    - Require **non-refundable or deposit-based bookings** to reduce no-shows.
    - Limit reliance on this group during peak seasons due to high cancellation volatility.

- **`Cluster 1` - Committed, High-Value Guests**
    - Promote **early bird premium packages** and **customisable booking experiences**.
    - Offer **loyalty incentives** to secure return visits and capitalise on early planning.
    - Focus on **value-added services** rather than discounts (e.g., early check-in, late checkout).
    - ***Working based off my own assumptions*** that bigger room types are useable as event space. If that holds true, ***advertising catering & event management services*** would definately be an ideal move.


## Key Difference Between Room Type 1 & 4 customers
While Room Type 1 featured **moderately price-sensitive leisure travelers**, **flexible weekday guests**, and **premium travellers** Room Type 4 stands out due to the **extreme polarisation between highly volatile bookers and high-commitment planners**.

---

## Closing Note

This analysis highlights how price elasticity modeling and customer segmentation can inform **targeted positioning and dynamic pricing strategies** beyond mere revenue optimisation. By decoding behavioral demand drivers, hospitality businesses can craft more meaningful, data-driven campaigns and service experiences.

View full analysis on: [GitHub](https://github.com/munas-git/hotel_booking_price_elasticity_modeling_and_customer_segmentation/tree/main) | [Kaggle Notebook](https://www.kaggle.com/code/munaee/price-elasticity-modeling-guest-segmentation)

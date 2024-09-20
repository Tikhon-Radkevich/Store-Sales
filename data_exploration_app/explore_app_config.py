data_description = {
    "train": """
        <p>This file contains the historical sales data, which covers sales from 2013-01-01 to 2017-08-15.</p>
        <ul>
            <li>some info</li>
            <li>another info</li>
        </ul>
        """,
    "test": """
        This file contains the historical sales data, which covers sales from 2013-01-01 to 2017-08-15.
        * some info
        * another info
        """,
    "stores": """
        This file contains information about the stores, including:
        * store number
        * type of store
        * cluster number
        """,
    "transactions": """
        This file contains the number of transactions for each store on a daily basis.
        * store number
        * date
        * number of transactions
        """,
    "oil": """
        This file contains the daily oil prices.
        * date
        * oil price
        """,
    "holidays_events": """
        This file contains information about holidays and events.
        * date
        * type of holiday/event
        * locale
        * locale name
        * description
        * transferred
        """,
    "sample_submission": """
        This file contains the sample submissions format.
        * id
        * sales
        """,
}

profile_kwargs = {
    "dataset": {
        "description": "",
        "url": "https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data",
    },
    "vars": {"cat": {"n_obs": 54}},
    "plot": {"histogram": {"bins": 54}, "cat_freq": {"max_unique": 54}},
    "n_freq_table_max": 54,
    "n_obs_unique": 54,
    "n_extreme_obs": 54,
}

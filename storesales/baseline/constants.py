import numpy as np


FAMILY_GROUPS = [
    ("AUTOMOTIVE", "CLEANING", "DELI", "GROCERY I", "PERSONAL CARE", "PREPARED FOODS"),
    ("BEAUTY", "BEVERAGES", "BREAD/BAKERY", "DAIRY", "EGGS", "POULTRY"),
    (
        "CELEBRATION",
        "HOME CARE",
        "LADIESWEAR",
        "PET SUPPLIES",
        "PLAYERS AND ELECTRONICS",
        "PRODUCE",
    ),
    ("BABY CARE",),
    ("BOOKS",),
    ("FROZEN FOODS",),
    ("GROCERY II",),
    ("HARDWARE",),
    ("HOME AND KITCHEN I",),
    ("HOME AND KITCHEN II",),
    ("HOME APPLIANCES",),
    ("LAWN AND GARDEN",),
    ("LINGERIE",),
    ("LIQUOR,WINE,BEER",),
    ("MAGAZINES",),
    ("MEATS",),
    ("SCHOOL AND OFFICE SUPPLIES",),
    ("SEAFOOD",),
]

N_STORES = 54
STORES = np.arange(1, N_STORES + 1)

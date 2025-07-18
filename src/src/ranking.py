def rank_candidates(features_with_scores: list[dict]) -> list[dict]:
    # مرتب‌سازی بر اساس سمیت کمتر و نزدیک بودن LogP به 2.0 به عنوان معیار ایده‌آل
    return sorted(features_with_scores, key=lambda x: (x['Toxicity'], abs(x['LogP']-2.0)))

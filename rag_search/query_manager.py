import importlib

class QueryManager:
    # def __init__(self):
        # # 映射数据库决策到相应的处理类路径
        # self.handler_map = {
        #     "linear regression": "transaction.LinearRegressionModel",
        #     "gaussianNB": "weather.WeatherModel",
        #     "Gaussian Naive Bayes": "weather.WeatherModel",
        #     "Artist Recommendation Model": "music_reco.ArtistRecommender",
        #     "transactions": "transaction.LinearRegressionModel",  # 假设数据库决策为transactions时对应的处理类
        #     "weather": "weather.WeatherModel",
        #     "spotify": "music_reco.ArtistRecommender",
        #     "Neural network": "test.Recommender"
        # }

    def get_handler(self, simplified_model_name, intent, database_decision=None):
        # 自动映射含有 "recommender" 的模型名称到 test.Recommender 类
        if "rec" in simplified_model_name.lower() or "neural" in simplified_model_name.lower():
            return self.load_class("test.Recommender")
        
        if "linear" in simplified_model_name.lower():
            return self.load_class("Linear_regression.LinearRegressionModel")
        
        # Default handler based on model or database decision
        handler_path = self.handler_map.get(simplified_model_name) or self.handler_map.get(database_decision)
        if handler_path:
            return self.load_class(handler_path)
        raise ValueError(f"Handler for model {simplified_model_name} or database {database_decision} not found.")

    def load_class(self, path):
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)()
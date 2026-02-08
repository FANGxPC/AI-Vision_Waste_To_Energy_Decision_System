import requests

def get_weather_context(api_key='5734a5b5abfbbfbaa2c050596331ef46', city="Vellore"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    default_weather = {"humidity": 50, "rain_1h": 0, "is_monsoon": False}
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            humidity = data['main'].get('humidity', 50)
            rain_1h = data.get('rain', {}).get('1h', 0)
            
            return {
                "humidity": humidity,
                "rain_1h": rain_1h,
                "is_monsoon": humidity > 80 or rain_1h > 0
            }
        else:
            print(f"Weather API Warning: {data.get('message', 'Unknown Error')}")
            return default_weather
            
    except Exception as e:
        print(f"Weather API Connection Error: {e}")
        return default_weather
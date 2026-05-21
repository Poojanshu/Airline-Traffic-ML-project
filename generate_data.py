import os
import pandas as pd
import numpy as np
from datetime import datetime

def generate_aviation_dataset():
    print("Generating U.S. Commercial Aviation Industry Metrics (2002 - 2017)...")
    
    # Define time period: Oct 2002 to Mar 2017 (174 months)
    dates = pd.date_range(start="2002-10-01", end="2017-03-01", freq="MS")
    
    # 4 major airlines and their base metrics profile
    # average_load_factor, trend_growth_rate, volatility
    airlines = {
        "DL": {"name": "Delta Air Lines", "avg_lf": 0.82, "growth": 0.001, "vol": 0.02, "avg_dist": 1050},
        "UA": {"name": "United Airlines", "avg_lf": 0.80, "growth": 0.0008, "vol": 0.025, "avg_dist": 1200},
        "AA": {"name": "American Airlines", "avg_lf": 0.79, "growth": 0.0009, "vol": 0.022, "avg_dist": 1100},
        "WN": {"name": "Southwest Airlines", "avg_lf": 0.77, "growth": 0.0012, "vol": 0.03, "avg_dist": 800}
    }
    
    # 4 major airports and their base size scaling factor
    airports = {
        "ATL": {"name": "Hartsfield-Jackson Atlanta", "scale": 1.4},
        "ORD": {"name": "Chicago O'Hare", "scale": 1.2},
        "LAX": {"name": "Los Angeles International", "scale": 1.1},
        "JFK": {"name": "John F. Kennedy International", "scale": 0.9}
    }
    
    # Seasonality multipliers (monthly)
    # Summer (Jun, Jul, Aug) and Dec are high traffic; Jan, Feb are low traffic
    seasonality = {
        1: 0.85,  # Jan (dip)
        2: 0.88,  # Feb (dip)
        3: 1.02,  # Mar
        4: 1.00,  # Apr
        5: 1.05,  # May
        6: 1.15,  # Jun (peak)
        7: 1.20,  # Jul (peak)
        8: 1.18,  # Aug (peak)
        9: 0.95,  # Sep
        10: 1.00, # Oct
        11: 0.98, # Nov
        12: 1.10  # Dec (holiday peak)
    }
    
    data = []
    np.random.seed(42)  # For reproducible dataset generation
    
    for dt in dates:
        month_str = dt.strftime("%Y-%m")
        m = dt.month
        y = dt.year
        
        # Calculate year-based index for trend
        months_since_start = (dt.year - 2002) * 12 + (dt.month - 10)
        
        for airline, air_meta in airlines.items():
            for airport, port_meta in airports.items():
                # Base passengers for this combination
                base_passengers = 150000 * air_meta["avg_dist"] / 1000 * port_meta["scale"]
                
                # Apply long-term upward trend
                trend = 1 + (months_since_start * air_meta["growth"])
                
                # Apply seasonality
                seas = seasonality[m]
                
                # Add random noise/variability (recessions, fuel shocks, etc.)
                # E.g., Great Recession shock in 2008-2009
                recession_factor = 0.92 if (y == 2008 or y == 2009) else 1.0
                
                noise = np.random.normal(0, air_meta["vol"])
                
                # Final passengers calculation
                passengers = int(base_passengers * trend * seas * recession_factor * (1 + noise))
                
                # Flights are proportional to passengers (average of 140 passengers per flight)
                pass_per_flight = np.random.normal(135, 8)
                flights = int(passengers / pass_per_flight)
                
                # Available Seat Miles (ASM) = Total seats available * distance
                # Simulates available capacity. Southwest has lower average seat capacity per plane.
                seats_per_flight = 160 if airline == "WN" else 185
                avg_distance = air_meta["avg_dist"] * np.random.normal(1.0, 0.02)
                asm = int(flights * seats_per_flight * avg_distance)
                
                # Load Factor (Efficiency metric between 60% and 92%)
                # High seasonality: airlines are more efficient in summer
                base_lf = air_meta["avg_lf"] * recession_factor
                # Seasonality effect on efficiency
                lf_seas = 0.05 * (seas - 1.0)
                load_factor = base_lf + lf_seas + np.random.normal(0, 0.015)
                # Clip load factor between realistic ranges
                load_factor = max(0.60, min(0.95, load_factor))
                
                # Revenue Passenger Miles (RPM) = ASM * Load_Factor
                rpm = int(asm * load_factor)
                
                # Recalculate passengers slightly to match RPM precisely (RPM = Passengers * Distance)
                # RPM / avg_distance = Passengers
                passengers = int(rpm / avg_distance)
                
                data.append({
                    "Month": month_str,
                    "Year": y,
                    "Month_Num": m,
                    "Airline": airline,
                    "Airport": airport,
                    "Passengers": passengers,
                    "Flights": flights,
                    "ASM": asm,
                    "RPM": rpm,
                    "Load_Factor": round(load_factor, 4)
                })
                
    df = pd.DataFrame(data)
    
    # Create target directories if they don't exist
    os.makedirs(os.path.dirname("C:/Users/hp/.gemini/antigravity/scratch/us_aviation_project/data/"), exist_ok=True)
    os.makedirs("C:/Users/hp/.gemini/antigravity/scratch/us_aviation_project/plots/", exist_ok=True)
    
    # Save to CSV
    csv_path = "C:/Users/hp/.gemini/antigravity/scratch/us_aviation_project/data/aviation_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated successfully! Saved {len(df)} rows to '{csv_path}'.")
    
if __name__ == "__main__":
    generate_aviation_dataset()

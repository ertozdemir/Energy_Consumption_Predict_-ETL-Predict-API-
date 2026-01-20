import pandas as pd 
import numpy as np 
import os 
import joblib 
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_data():
    df_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train_energy_data.csv'))
    df_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test_energy_data.csv'))
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    return df


def preprocess_data(df):
    # 1 foot = 0.092903 square meters
    df.rename(columns={'Building Type' : 'Building_Type',
    'Square Footage': 'Square_Footage', 
    'Number of Occupants': 'Number_of_Occupants',
    'Appliances Used': 'Appliances_Used',
    'Average Temperature' : 'Average_Temperature',
    'Day of Week' : 'Day_of_Week',
    'Energy Consumption': 'Energy_Consumption'}, inplace=True)

    df['Square_Meters'] = df['Square_Footage'] * 0.092903

    df.drop('Square_Footage', axis=1, inplace=True)
    return df



def write_db(df):

    db_path = os.path.join(BASE_DIR, 'database', 'energy_consumption.db')
    engine = create_engine(f'sqlite:///{db_path}')
    base = declarative_base()
    
    class EnergyData(base):
        __tablename__ = 'energy_data'
        id = Column(Integer, primary_key=True)
        building_type = Column(String)
        square_meters = Column(Integer)
        number_occupants = Column(Integer)
        appliances_used = Column(String)
        day_of_week = Column(String)
        energy_consumption = Column(Float)

    base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    for index, row in df.iterrows():
        energy_data = EnergyData(
            building_type = row['Building_Type'],
            square_meters = row['Square_Meters'],
            number_occupants = row['Number_of_Occupants'],
            appliances_used = row['Appliances_Used'],
            day_of_week = row['Day_of_Week'],
            energy_consumption = row['Energy_Consumption']
        )
        session.add(energy_data)

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

if __name__ == '__main__':
    df = load_data()
    print('Data loaded')
    df = preprocess_data(df)
    print('Data preprocessed')
    write_db(df)
    print('Data written to database')
        

    
    






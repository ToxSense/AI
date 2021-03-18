#!/usr/bin/python

def aqius(val, type):
    index = 0
    if (val >= 0):
        if (type == 'PM10'):
            if (int(val) <= 54):
                index = calculate_aqi_us(50, 0, 54, 0, int(val))
            elif(int(val) <= 154):
                index = calculate_aqi_us(100, 51, 154, 55, int(val))
            elif(int(val) <= 254):
                index = calculate_aqi_us(150, 101, 254, 155, int(val))
            elif(int(val) <= 354):
                index = calculate_aqi_us(200, 151, 354, 255, int(val))
            elif(int(val) <= 424):
                index = calculate_aqi_us(300, 201, 424, 355, int(val))
            elif(int(val) <= 504):
                index = calculate_aqi_us(400, 301, 504, 425, int(val))
            elif(int(val) <= 604):
                index = calculate_aqi_us(500, 401, 604, 505, int(val))
            else:
                index = 500
        if (type == 'PM25'):
            if (round(val, 1) <= 12):
                index = calculate_aqi_us(50, 0, 12, 0, round(val, 1))
            elif(round(val, 1) <= 35.4):
                index = calculate_aqi_us(100, 51, 35.4, 12.1, round(val, 1))
            elif(round(val, 1) <= 55.4):
                index = calculate_aqi_us(150, 101, 55.4, 35.5, round(val, 1))
            elif(round(val, 1) <= 150.4):
                index = calculate_aqi_us(200, 151, 150.4, 55.5, round(val, 1))
            elif(round(val, 1) <= 250.4):
                index = calculate_aqi_us(300, 201, 250.4, 150.5, round(val, 1))
            elif(round(val, 1) <= 350.4):
                index = calculate_aqi_us(400, 301, 350.4, 250.5, round(val, 1))
            elif(round(val, 1) <= 500.4):
                index = calculate_aqi_us(500, 401, 500.4, 350.5, round(val, 1))
            else:
                index = 500
    return index

def calculate_aqi_us(Ih, Il, Ch, Cl, C):
    return int((((Ih - Il) / (Ch - Cl)) * (C - Cl)) + Il)

def officialAQIus(data):
    p1 = aqius(data["PM10"], 'PM10');
    p2 = aqius(data["PM25"], 'PM25');
    if (p1 >= p2):
        return p1
    else:
        return p2
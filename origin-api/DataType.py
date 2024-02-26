from pydantic import BaseModel

class DataType(BaseModel):
    floor: float
    room_size: float
    unit_type_id: int
    district_id: int
    Airport_Rail_Link: int
    BTS_Silom_Line: int
    BTS_Sukhumvit_Line: int
    District: int
    Gold_Line: int
    Government: int
    Hospital: int
    International_School: int
    Market: int
    MRT_Blue_line: int
    Popular_Areas: int
    Road: int
    School: int
    Shopping_Mall: int
    Soi: int
    Super_Market: int
    University_College: int

class Trend(BaseModel):
    district_id: int
    rental_group: int
    price_group: int

class Year(BaseModel):
    year: int
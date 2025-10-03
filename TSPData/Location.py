import math


class Location:
    ''' Represents a geographical location with latitude and longitude. '''
    def __init__(self, latitude: float, longitude: float):
        ''' Initializes a Location object with latitude and longitude. '''
        self._latitude = latitude
        self._longitude = longitude

    @property
    def latitude(self) -> float:
        return self._latitude
    
    @property
    def latitude(self) -> float:
        return self._latitude

    @property
    def longitude(self) -> float:
        return self._longitude

    @staticmethod
    def to_geo_radians(ddmm: float) -> float:
        deg = int(ddmm)
        minutes = ddmm - deg
        return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0

    def to_geo(self):
        x = Location.to_geo_radians(self._latitude)
        y = Location.to_geo_radians(self._longitude)
        return Location(x, y)

    def get_geo_great_circle_distance(self, location: 'Location') -> int:
        # TSPLIB GEO great-circle distance on a sphere (radius 6378.388 km)
        q1 = math.cos(self._longitude - location._longitude)
        q2 = math.cos(self._latitude - location._latitude)
        q3 = math.cos(self._latitude + location._latitude)
        dij = 6378.388 * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
        return int(dij)  # integer truncation per TSPLIB (floor)

    def geo_distance(self, other: "Location") -> int:
        """Wrapper for TSPLIB GEO great-circle distance."""
        return self.get_geo_great_circle_distance(other)

    def get_pseudo_euclidean_distance(self, location: "Location") -> int:
        # TSPLIB Pseudo-Euclidean distance
        xd = self._latitude - location.latitude
        yd = self._longitude - location.longitude
        rij = math.sqrt((xd * xd + yd * yd) / 10.0)
        tij = round(rij)
        return tij + (1 if tij < rij else 0)  # integer rounding per TSPLIB

    # Existing get_euclidean can remain or delegate (optional)
    def get_euclidean(self, location: "Location") -> float:
        dx = self._latitude - location.latitude
        dy = self._longitude - location.longitude
        return math.sqrt(dx * dx + dy * dy)

    def __str__(self):
        return f"Location {{X = {self.latitude}, Y = {self.longitude}}}"

# -*- coding: utf-8 -*-

import datetime

class WeekCalculator(object):
    
    def __init__(self):
        self.week_dict = {}
        self.cumulative_week_dict = {}
    
    def calc_weeks_in_year(self, y):
        week = 1
        d = 31
        while week == 1:
            date = "{0}-12-{1}".format(y, d)
            isodate = datetime.datetime.strptime(date, "%Y-%m-%d").isocalendar()
            week = isodate[1]
            d -= 1
        return isodate[1]
    
    def weeks_in_year(self, y):
        if y in self.week_dict:
            return self.week_dict[y]
        w = self.calc_weeks_in_year(y)
        self.week_dict[y] = w
        return w
    
    def weeks_before_year(self, y):
        if y in self.cumulative_week_dict:
            return self.cumulative_week_dict[y]
        year = y - 1
        weeks = 0
        while year >= 2003:
            weeks += self.weeks_in_year(year)
            year -= 1
        self.cumulative_week_dict[y] = weeks
        return weeks
    
    def calc_week(self, date):
        isodate = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").isocalendar()
        week = isodate[1] + self.weeks_before_year(isodate[0])
        return week
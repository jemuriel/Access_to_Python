class Terminal:
    def __init__(self, name):
        self.name = name
        self.wagon_inventory = {}

class Train_Leg:

    def __init__(self, origin, destination, departure_day, departure_time,
                 arrival_day, arrival_time):

        self.leg_origin_terminal = origin
        self.leg_destination_terminal = destination
        self.departure_day = departure_day
        self.departure_time = departure_time
        self.arrival_day = arrival_day
        self.arrival_time = arrival_time

        self.dep_min_num = self.time_to_minutes(self.departure_day, self.departure_time)
        self.arr_min_num = self.time_to_minutes(self.arrival_day, self.arrival_time)

        self.wagon_configuration={} # PLATFORM: WAGON OBJECT

    # def time_to_minutes(self, day, time_str):
    #     days = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
    #     day_minutes = days[day] * 24 * 60
    #     h, m = map(int, time_str.split(':'))
    #     return day_minutes + h * 60 + m

    def time_to_minutes(self, day, time_obj):
        days = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
        day_minutes = days[day] * 24 * 60
        # Use hour and minute attributes of the time object
        return day_minutes + time_obj.hour * 60 + time_obj.minute


    def __str__(self):
        return f'{self.leg_origin_terminal}-{self.leg_destination_terminal}'

class Wagon:
    def __init__(self, type, size, num_wagons, platform, num_platforms, family, all_equal, configuration):
        self.type=type
        self.size=size
        self.num_wagons=num_wagons
        self.platform = platform
        self.num_platforms = num_platforms
        self.family=family
        self.total_platforms = self.num_wagons*self.num_platforms
        self.all_equal = all_equal #boolean to indicate if it's an odd configuration e.g. 40-48-48-48-40
        self.configuration = configuration.split('-')

    def __str__(self):
        return f'Type: {self.type} - Family: {self.platform} - Platforms: {self.total_platforms}'
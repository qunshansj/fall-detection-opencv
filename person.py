python
class Person(object):
    """Person"""
    amount = 0

    def __init__(self, x, y, w, h, movementMaximum, movementMinimum, movementTime):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.movementTime = movementTime
        self.movementMaximum = movementMaximum
        self.movementMinimum = movementMinimum
        self.lastmoveTime = 0
        self.alert = 0
        self.alarmReported = 0
        self.lastseenTime = 0
        self.remove = 0
        Person.amount += 1
        if Person.amount > 1000:  # If person amount > 1000, set amount to 0
            Person.amount = 0
        self.id = Person.amount

    def samePerson(self, x, y, w, h):  # To Check if its the same person in the frame
        same = 0
        if x+self.movementMaximum > self.x and x-self.movementMaximum < self.x:
            if y+self.movementMaximum > self.y and y-self.movementMaximum < self.y:
                same = 1
        return same

    def editPerson(self, x, y, w, h):  # To Check 
        if abs(x-self.x) > self.movementMinimum or abs(y-self.y) > self.movementMinimum or abs(w-self.w) > self.movementMinimum or abs(h-self.h) > self.movementMinimum:
            self.lastmoveTime = 0
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.lastseenTime = 0

    def getId(self):   # To get the id of the person
        return self.id

    def tick(self):
        self.lastmoveTime += 1  # Lastmovetime
        self.lastseenTime += 1  # Lastseentime
        if self.lastmoveTime > self.movementTime:
            self.alert = 1
        if self.lastseenTime > 4: # how many frames ago last seen
            self.remove = 1

    def getAlert(self):  # Raise an alert function
        return self.alert

    def getRemove(self):  
        return self.remove


class Persons:
    def __init__(self, movementMaximum, movementMinimum, movementTime):
        self.persons = []  # Making an empty list of persons
        self.movementMaximum = movementMaximum
        self.movementMinimum = movementMinimum
        self.movementTime = movementTime
        Person.amount = 0   # Initial count of person

    def addPerson(self, x, y, w, h):
        person = self.familiarPerson(x, y, w, h)
        if person:  # if its the same person edit the coordinates and w, h
            person.editPerson(x, y, w, h)
            return person
        else:       # Else append the person to the list
            person = Person(x ,y ,w ,h , self.movementMaximum, self.movementMinimum, self.movementTime)
            self.persons.append(person)
            return person

    def familiarPerson(self, x, y, w, h):  # To check if its the same person
        for person in self.persons:
            if person.samePerson(x, y, w, h):
                return person
        return None

    def tick(self):
        for person in self.persons:
            person.tick()
            if person.getRemove():
                self.persons.remove(person)

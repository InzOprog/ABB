from math import ceil

with open("routes.txt", 'w') as file:
    for i in range(2000):
        file.write("<vehicle id='{id}' route='r{r_id}' depart='{dt}'/>\n".format(
            id=i, r_id=(i % 10), dt=(ceil((i+1)/5.0) * 5))
        )

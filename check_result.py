from app.retrieval.search import RetrievalResult, search
import inspect

# Print the actual class definition
print(inspect.getsource(RetrievalResult))
print()

# Print all attributes on a live result
r = search('robotics lab', lang='en')
print('type:', type(r))
print('vars:', vars(r))

src_ip = "192"
src_port = "1"
dst_ip = "168"
dst_port = "2"

features = ['Benign','Malignant', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7']


json_string = f"""{{"src_ip": {str(src_ip)}, "src_port": {str(src_port)}, "dst_ip": {str(dst_ip)}, "dst_port": {str(dst_port)}, "result": {features[0]}, "probability": {[[1]]}}}"""
print("JSON"+str(json_string))
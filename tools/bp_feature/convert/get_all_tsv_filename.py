
import os
import json

def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return



if __name__ == '__main__':
    print()


    tsv_file_list=list(os.listdir("/multi_modal/data/tsv_features"))

    write_json("tsv_filename_list.json", tsv_file_list)






# -*- coding: utf-8 -*-

import click
import unetsl.management
import pathlib
import itertools
    
def getGpusFromParameter(p):
    return [int(i) for i in p.split(",")]
    

@click.command()
@click.argument("model_folders", type=click.Path(exists=True), nargs=-1)
@click.option("-s", "--sample_folder", type=click.Path(exists=True))
@click.option("-p", "--prediction_folder", type=click.Path(exists=True))
@click.option("-t", "--prediction_type", default="image")
@click.option("-g", "--available_gpu", envvar='CUDA_VISIBLE_DEVICES')
@click.option("-o", "--output_index", default=-1, type=click.INT)
@click.option("-D", "--extended_options", multiple=True )
@click.option("-m", "multi_channel", is_flag=True)
def main(
        model_folders, 
        sample_folder, 
        prediction_folder, 
        prediction_type, 
        available_gpu, 
        output_index, 
        extended_options, 
        multi_channel
        ):
    models = unetsl.management.getModels(model_folders)
    if prediction_type == "image":
        sample_folders = [pathlib.Path(sample_folder)]
    else:
        sample_folders = [d for d in pathlib.Path(sample_folder).iterdir() if d.is_dir ]
    gpus = getGpusFromParameter(available_gpu)
    pool = unetsl.management.getGpuPool(gpus)
    clog = open("predictions-cmds.log.txt", 'w')
    for model, sample_folder in itertools.product(models, sample_folders):
        print("model %s :: sample folder %s"%(model, sample_folder))
        model_ctime = model.stat().st_ctime
        
        output_folder = pathlib.Path(prediction_folder)
        
        if not output_folder.exists():
            output_folder.mkdir()
        elif not output_folder.is_dir():
            raise Exception("output folder is a file!")
        
        for img in sample_folder.iterdir():
            if not img.name.endswith("tif"):
                continue
            output_name = unetsl.management.getOutputName(model.name, img.name, output_index)
                
            op_image = pathlib.Path(output_folder, output_name)
            if op_image.exists() and model_ctime <= op_image.stat().st_ctime:
                continue
            else:
                print("\t\t", img)
                cmd = ["predict_image", str(model), str(img), str(op_image), "-b" ]
                
                if "-sn" in model.name:
                    cmd.append("-D")
                    cmd.append("normalize samples=true")
                if "-csc" in model.name:
                    cmd.append("-D")
                    cmd.append("output index=%d"%output_index)
                elif "-crb" in model.name:
                    multi_channel=True
                    
                for eo in extended_options:
                    cmd.append("-D");
                    cmd.append(eo)

                if multi_channel:
                    cmd.append("-m")

                clog.write("#executing::\n %s\n"%(" ".join(cmd)))
                pool.post(cmd)
    clog.close()
    pool.shutdown()
    plog = open("gpu-pool.log", 'w')

    pool.save_log(plog)
    plog.close()
        
if __name__=="__main__":
    main()
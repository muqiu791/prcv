import glob
import json
import os

RGBTmodel = 'fuseIRVIS'  # 'infrared' 'visible' 'fuseIRVIS' 'fuseIR_UnaffVIS'
def main():
    root_path = 'E:/UNIFusion-main/outputs/Anti-UAV-RGBT/test'

    subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]

    all_data = {}  # This will hold all the data

    # 遍历每个子文件夹
    for subfolder in subfolders:
        subsubfolder = os.path.join(subfolder, RGBTmodel)
        print('folr_path', subsubfolder)

        video_dir = os.path.basename(subfolder)
        print('video_dir', video_dir)

        filenames = [(video_dir+'/fuseIRVIS/'+os.path.basename(f)) for f in glob.glob(os.path.join(subsubfolder, '*.jpg'))]
        print('filenames', len(filenames))

        print('filenames', filenames)

        gt_json_file = os.path.join(subfolder, f'{RGBTmodel}.json')
        with open(gt_json_file, 'r') as json_file:
            data = json.load(json_file)

            exist_values = data.get("exist", [])
            gt_rect_values = data.get("gt_rect", [])

            for i in range(len(exist_values)):
                if exist_values[i] == 1:
                    init_rect = gt_rect_values[i]
                    print('init_rect', init_rect)
                    break
                else:
                    continue

        # Store data for this subfolder
        all_data[video_dir] = {
            "video_dir": video_dir,
            "img_names": filenames,
            "gt_rect": gt_rect_values,
            "init_rect": init_rect
        }

    # Write all data to a single JSON file
    with open(os.path.join(root_path, f'{RGBTmodel}_all_output.json'), 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

    print(f'All data written to {RGBTmodel}_all_output.json')


if __name__ == '__main__':
    main()

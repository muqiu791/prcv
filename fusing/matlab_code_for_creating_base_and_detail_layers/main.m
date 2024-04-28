% Specify the root directory containing subfolders
current_dir = fileparts(mfilename('fullpath'));

dataset_root = '../Anti-UAV-RGBT/test';
outputRoot = '../tracking/Myoutput/Anti-UAV-RGBT/test';

% 计算相对路径
dataset_root = fullfile(current_dir, dataset_root);
outputRoot = fullfile(current_dir, outputRoot);



% Get a list of subfolders in the specified root directory
subfolders = dir(outputRoot);
subfolders = subfolders([subfolders.isdir] & ~strcmp({subfolders.name}, '.') & ~strcmp({subfolders.name}, '..'));

% Iterate over each subfolder
for folderIdx = 1:length(subfolders)
    subfolderName = subfolders(folderIdx).name;

    % Construct the full path to the subfolder
    % subfolderPath = fullfile(dataset_root, subfolderName, 'visible');
    subfolderPath = fullfile(outputRoot, subfolderName, 'affine_vis');
    % 'affine_vis'
    
    % Now you can perform operations specific to each subfolder
    % For example, display the subfolder name
    disp(['Processing subfolder: ', subfolderName]);

    % 如果你想在子文件夹中处理文件，你可以再次使用dir
    filesInSubfolder = dir(fullfile(subfolderPath, '*.jpg'));

    % 显示匹配到的文件
    disp('匹配到的文件:');
    disp({filesInSubfolder.name});
    for fileIdx = 1:length(filesInSubfolder)

        fileName = filesInSubfolder(fileIdx).name;

        % Construct the full path to the file
        filePath = fullfile(subfolderPath, fileName);
        disp(['filePath : ', filePath]);

        %% Read the image
        VIS = im2double(imread(filePath));
        VIS = imresize(VIS, [512, 640]);  % Resize the image to H x W
        IR = im2double(imread(filePath)); % Assuming IR images are in the same folder


        %% Do the job
        alpha_t = 0.001;
        N_iter = 3;
        tic
        [VISBase, ~] = muGIF(VIS, VIS, alpha_t, 0, N_iter);
        [IRBase, ~] = muGIF(IR, IR, alpha_t, 0, N_iter);
        toc

        VISDetail = VIS - VISBase;
        IRDetail = IR - IRBase;

        VISDetail = VISDetail + mean(mean(VIS));
        IRDetail = IRDetail + mean(mean(IR));

        % Assuming you have created a folder named 'Output' to store the results
        % 拼接outputRoot和最后一级子文件夹名得到outputFolder
        IRoutputFolder = fullfile(outputRoot, subfolderName,'IR');
        VISoutputFolder = fullfile(outputRoot, subfolderName,'VIS');
        aff_visoutputFolder = fullfile(outputRoot, subfolderName,'aff_VIS');
        
        % 检查是否存在，不存在则创建
        if ~exist(IRoutputFolder, 'dir')
            mkdir(IRoutputFolder);
        end

        if ~exist(VISoutputFolder, 'dir')
            mkdir(VISoutputFolder);
        end
        
        if ~exist(aff_visoutputFolder, 'dir')
            mkdir(aff_visoutputFolder);
        end
        
        aff_VISBase_filename = sprintf('%s/VISBase%04d.jpg', aff_visoutputFolder, fileIdx - 1);
        aff_VISDetail_filename = sprintf('%s/VISDetail%04d.jpg', aff_visoutputFolder, fileIdx - 1);
        % Writing output images
        imwrite(IRBase, [IRoutputFolder, '/IRBase',num2str(fileIdx-1), '.jpg']);
        imwrite(IRDetail, [IRoutputFolder,'/IRDetail',num2str(fileIdx-1), '.jpg']);
        imwrite(VISBase, aff_VISBase_filename);
        imwrite(VISDetail, aff_VISDetail_filename);

        disp(['Process:(', num2str(fileIdx), '/', num2str(length(filesInSubfolder)), ')']);
    end

end

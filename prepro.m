clear all,close all

path ='test\';

Files = dir(strcat(path,'*.tif'));
r_rand = randi([-15 15],10000,1);
for i = 1:length(Files)
    gg = imread(strcat(path,Files(i).name));
    k = double(gg);
    h = k/max(max(k));
    h = h*255;
    h1=uint8(h);
    h = adapthisteq(h1);
    
    f=Files(i).name;
    imwrite(h,['rot\' path(1:end) f(1:end-4) '.png'])
    
    %%Uncomment the following to generate augmented dataset including
    %%rotations
%     ir = imrotate(h,180,'nearest','crop');
%     imwrite(ir,['rot\' path(1:end) f(1:end-4) 'r1.png'])
%     ir = imrotate(h,30,'nearest','crop');
%     imwrite(ir,['rot\' path(1:end) f(1:end-4) 'r2.png'])
%     ir = imrotate(h,-30,'nearest','crop');
%     imwrite(ir,['rot\' path(1:end) f(1:end-4) 'r3.png'])
%     ir = imrotate(h,15,'nearest','crop');
%     imwrite(ir,['rot\' path(1:end) f(1:end-4) 'r4.png'])
%     ir = imrotate(h,-15,'nearest','crop');
%     imwrite(ir,['rot\' path(1:end) f(1:end-4) 'r5.png'])
%     ir = imrotate(h,10,'nearest','crop');
%     imwrite(ir,['rot\' path(1:end) f(1:end-4) 'r6.png'])
%     ir = imrotate(h,-10,'nearest','crop');
%     imwrite(ir,['rot\' path(1:end) f(1:end-4) 'r7.png'])
   
end

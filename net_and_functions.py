class FishFinDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fish_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fish_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.fish_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.fish_frame.iloc[idx, 1]
        label2 = self.fish_frame.iloc[idx,2]
        convert_strings = {'Has_fin':0, 'No_fin':1, 'Cannot_see':2}
        convert_strings2 = {'No Fungi': 0, 'Slight Fungi':1, 'Severe Fungi':2}

        sample = {'image': image, 'label': np.array([convert_strings[label]]), 'label2': np.array([convert_strings2[label2]])}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, label2 = sample['image'], sample['label'], sample['label2']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        label = label.reshape((-1,))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'label2':torch.from_numpy(label2)}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, label2 = sample['image'], sample['label'], sample['label2']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        

        return {'image': img, 'label': label, 'label2': label2}


class MultiHeadResNet(torch.nn.Module):
    def __init__(self,net):
        super().__init__()
        self.conv1 = list(net.children())[0]
        self.bn1 = list(net.children())[1]
        self.relu = list(net.children())[2]
        self.maxpool = list(net.children())[3]
        self.layer1 = list(net.children())[4]
        self.layer2 = list(net.children())[5]
        self.layer3 = list(net.children())[6]
        self.layer4 = list(net.children())[7]
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512,3)
        self.fc2 = nn.Linear(512,3)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1,x2



#####################################

#KOLLA IGENOM CUSTOMKLASSEN 


#####################################

def my_collate(batch):
    data = [item['image'] for item in batch]
    target = [item['label'] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
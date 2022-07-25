from torchvision import transforms

TRAIN_TRANSFORMS = {
    'MNIST': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    'CIFAR10': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'Diabetic': transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomSizedCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
}

TEST_TRANSFORMS = {
    'MNIST': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    'CIFAR10': transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'Diabetic': transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
}


def get_transforms(dataset):
    return {
        'train': TRAIN_TRANSFORMS[dataset],
        'test': TEST_TRANSFORMS[dataset]
    }


def get_transforms_with_rotation(dataset, rotation_angle, train_rotation=True, test_rotation=True):
    if dataset == 'CIFAR10':
        data_transforms = dict()
        if train_rotation:
            data_transforms['train'] = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(rotation_angle),
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            data_transforms['train'] = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        if test_rotation:
            data_transforms['test'] = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(rotation_angle),
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        else:
            data_transforms['test'] = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        return data_transforms
    else:
        raise NotImplementedError

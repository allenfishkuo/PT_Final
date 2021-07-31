# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:30:28 2020

@author: allen
"""

import torch
def trans():
    model = torch.load('2013-2016NewNew.pkl')
    model.eval()
    torch.save(model.state_dict(),'2013-2016NewNewNew.pth' )
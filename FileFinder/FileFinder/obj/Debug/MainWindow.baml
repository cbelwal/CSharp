   M S B A M L   `   `   `  ����E  AFileFinder, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null'clr-namespace:FileFinder
FileFinder  5   	   ;"clr-namespace:FileFinder.ResourcesFileFinder.Resources  5   	       FileFinder.MainWindow   5   	   \ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5   	   R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   *localclr-namespace:FileFinder     5   	   7localRes"clr-namespace:FileFinder.Resources     5   	   *   $FileFinder.Resources.StringResources    Title � Title# Z  5	   	   $	��450��6;   $	��725��6H    � 
ResizeMode$ NoResize=�6T    � WindowStartupLocation$ CenterScreen=�6j    � DataContext 5
         FileFinder.ViewModelMain  5   
   .��5      �  � RowDefinitions 5   
   �� 5      $
��Auto �6   �� 5      $
��Auto �6   �� 5      $��* �6   �� 5      $
��Auto �6   .+�5   
   � $��0��6    � ColumnDefinitions 5      �� 5      $
��Auto �6#   �� 5      $��* �6#   �� 5      $
��Auto �6#   .+�5      �� $��0��6   $��0��6!      Folder#��Z 61   $
��Left=�5      $��5,5,5,5q�6+   $	��Top=�6<   �� 5      $��0��6   $��1��6#   $��Stretch=�5      $��5,5,5,5q�6.   $	��Top=�6?   ��5   3   �� *
FolderName+ �� Mode$ TwoWay=�5   ?   �� 5       $��0��6   $��2��6"   	   Browse#��Z	 62   $
��Left=�5!      $��5,5,5,5q�6+   $	��Top=�6<   ��5"      �� *BrowseCommand+5#      � 5$   
   $��1��6    5%      �� 5&      $
��Auto �6#   �� 5'      $	��53* �6#   �� 5(      $
��130* �6#   �� 5)      $
��Auto �6#   .+�5+      �� $��0��6   $��0��6!   
   Search#��Z
 61   $
��Left=�5,      $��5,5,0,0q�6+   $	��Top=�6<   �� 5-      $��0��6   $��1��6#   $��Stretch=�5.      $��5,5,5,0q�6.   $	��Top=�6?   $��2��6W   ��5-   3   �� *
SearchText+$ TwoWay=�5.   W   �� 5/      -   5        $��btnStart��5/      $��0��6&   $��3��63      Start#��Z 6C   $
��Left=�50      $��5,5,5,0q�6+   $	��Top=�6<   ��51      �� *StartCommand+53      �� 54   
   -   5        $��listBoxFoundFiles��55      $��2��54      $��Stretch=�6   $��5,5,5,5q�56      $ �� IsSynchronizedWithCurrentItem$
 TrueY�6#   ��55   +   �� *
FoundFiles+$ OneWay=���57      �� 58       �� 
TargetType" ��6    �� BasedOn# [B��6:   .�59      -   5        5:      5<      �� 5>   
   $��3��6   $��Stretch=�5?      ��5>      �� *StatusMessage+$ OneWay=�5?      5A      5B      
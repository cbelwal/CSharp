using System;
using System.Windows.Input;

namespace FileFinder
{
    public class RelayCommand : ICommand
    {
        Func<object, bool> _canExecute;
        Action<object> _execute;

        public bool CanExecute(object parameter)
        {
            if (_canExecute != null)
                return _canExecute(parameter);
            else return true;
        }

        public event EventHandler CanExecuteChanged;

        public void Execute(object parameter)
        {
            _execute(parameter);
        }

        public RelayCommand(Action<object> executeMethod)
        {
            _execute = executeMethod;
        }
    }
}

class ServiceLocator:
    _services = {}  # Class-level attribute to store registered services

    @staticmethod
    def register_service(name, service):
        """
        Register a service in the service container.

        Parameters:
        name (str): Name of the service.
        service: The service object to register.
        """
        ServiceLocator._services[name] = service  # Updates the class-level dictionary

    @staticmethod
    def get_service(name):
        """
        Retrieve a registered service by name.

        Parameters:
        name (str): Name of the service to retrieve.
        
        Returns:
        The registered service object.
        """
        return ServiceLocator._services.get(name)
